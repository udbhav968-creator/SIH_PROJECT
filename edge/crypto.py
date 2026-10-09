"""
Encrypted, authenticated packets between a bus and the server.

Each event leaves the bus as one small JSON packet sealed with AES-256-GCM:

    {"v": 1, "bus": "<bus id>", "ep": "<queue epoch>", "seq": <n>, "nonce": "<base64 12 bytes>",
     "ct": "<base64 ciphertext+tag>"}

    confidentiality  only a holder of the fleet key can read the packet
    integrity        GCM's tag fails on any change to the ciphertext
    binding          bus id, queue epoch and sequence number are additional authenticated data, so a packet
                     cannot be re-labelled as coming from another bus or replayed under a new number
    replay           the server keeps the highest sequence number it accepted per bus and epoch and refuses
                     anything at or below it (pipeline/edge_ingest.py)
    epoch            a random id created with each queue database. A bus whose SD card is replaced starts
                     again at sequence 1 in a new epoch, so its new packets are not mistaken for replays

Bus ids are limited to letters, digits, '.', '_' and '-' (at most 40), because they are shown on the
operators' map.

Nonces are random (96 bits) under one fleet key: rotate the key well before 2^32 packets in total.

The same ciphertext is what the bus stores while it is offline (edge/store_forward.py), so packets are
encrypted at rest on the device too: a stolen SD card does not expose the queue without the key.

Key: ROAD_SHIELD_FLEET_KEY, either 64 hex characters / 44 base64 characters (32 raw bytes), or any other
string, which is stretched with PBKDF2-HMAC-SHA256 (200,000 rounds, fixed salt).

Per-bus keys: the server holds the fleet MASTER key; each bus is given only its own key,
bus_key(master, bus_id) = HMAC-SHA256(master, "road-shield-bus-key|" + bus_id), printed by
`python -m edge.provision <BUS-ID>` and set as ROAD_SHIELD_FLEET_KEY on that bus. A stolen bus therefore
exposes one bus's key, not the fleet's, and can be revoked on its own (pipeline/edge_ingest.py). The server
tries the bus's derived key first; packets sealed with the master key itself are still accepted while
ROAD_SHIELD_ALLOW_FLEET_KEY is not 0, so buses provisioned before per-bus keys keep working until they are
re-provisioned.
"""
import base64
import hashlib
import json
import os
import re

VERSION = 1
MAX_PLAINTEXT_BYTES = 1024
_SALT = b"road-shield-fleet-key-v1"
BUS_ID = re.compile(r"[A-Za-z0-9._-]{1,40}")       # used with fullmatch: "$" would let a trailing newline in
EPOCH = re.compile(r"[0-9a-f]{1,32}")


class PacketError(ValueError):
    pass


def load_key(value=None):
    """32-byte key from the argument or ROAD_SHIELD_FLEET_KEY; None when neither is set."""
    raw = value if value is not None else os.environ.get("ROAD_SHIELD_FLEET_KEY", "")
    if isinstance(raw, bytes):
        if len(raw) == 32:
            return raw
        raw = raw.decode("utf-8", "replace")
    raw = (raw or "").strip()
    if not raw:
        return None
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw)
        except ValueError:
            pass
    if len(raw) == 44:
        try:
            k = base64.b64decode(raw, validate=True)
            if len(k) == 32:
                return k
        except Exception:
            pass
    return hashlib.pbkdf2_hmac("sha256", raw.encode("utf-8"), _SALT, 200_000, dklen=32)


def new_key_hex():
    return os.urandom(32).hex()


def bus_key(master, bus_id):
    """The key one bus seals with: derived from the master key and the bus id, so it is useless for any other bus."""
    import hmac
    if not BUS_ID.fullmatch(str(bus_id)):
        raise PacketError("bus id must be 1-40 letters, digits, '.', '_' or '-'")
    return hmac.new(master, b"road-shield-bus-key|" + str(bus_id).encode("utf-8"), hashlib.sha256).digest()


def peek_bus(envelope):
    """The bus id a packet claims (authenticated only after decryption), or None."""
    try:
        b = str(envelope.get("bus", ""))
        return b if BUS_ID.fullmatch(b) else None
    except Exception:
        return None


def _aad(bus_id, epoch, seq):
    return f"road-shield|v{VERSION}|{bus_id}|{epoch}|{int(seq)}".encode("utf-8")


def check_ids(bus_id, epoch):
    if not BUS_ID.fullmatch(str(bus_id)):
        raise PacketError("bus id must be 1-40 letters, digits, '.', '_' or '-'")
    if not EPOCH.fullmatch(str(epoch)):
        raise PacketError("malformed epoch")


def pack(event, bus_id, seq, key, epoch="0"):
    """Seal one event dict. Raises PacketError when the compact JSON is over MAX_PLAINTEXT_BYTES."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    check_ids(bus_id, epoch)
    plain = json.dumps(event, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
    if len(plain) > MAX_PLAINTEXT_BYTES:
        raise PacketError(f"event is {len(plain)} bytes; the limit is {MAX_PLAINTEXT_BYTES}")
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plain, _aad(bus_id, epoch, seq))
    return {"v": VERSION, "bus": str(bus_id), "ep": str(epoch), "seq": int(seq),
            "nonce": base64.b64encode(nonce).decode("ascii"), "ct": base64.b64encode(ct).decode("ascii")}


def unpack(envelope, key):
    """(bus_id, epoch, seq, event); PacketError if it was altered, re-labelled or sealed with another key."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        if int(envelope.get("v", 0)) != VERSION:
            raise PacketError("unsupported packet version")
        bus_id, epoch, seq = str(envelope["bus"]), str(envelope.get("ep", "0")), int(envelope["seq"])
        check_ids(bus_id, epoch)
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ct = base64.b64decode(envelope["ct"], validate=True)
    except PacketError:
        raise
    except Exception as e:
        raise PacketError(f"malformed packet: {e}")
    if len(nonce) != 12:
        raise PacketError("malformed packet: nonce must be 12 bytes")
    try:
        plain = AESGCM(key).decrypt(nonce, ct, _aad(bus_id, epoch, seq))
    except InvalidTag:
        raise PacketError("authentication failed: wrong key, or the packet was altered")
    event = json.loads(plain.decode("utf-8"))
    if not isinstance(event, dict):
        raise PacketError("packet does not contain an event object")
    return bus_id, epoch, seq, event
