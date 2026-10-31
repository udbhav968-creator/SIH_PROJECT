"""Cryptographic signing and payload verification for edge packets."""
import base64
import hashlib
import json
import os
import re

VERSION = 1
MAX_PLAINTEXT_BYTES = 1024
_SALT = b"road-shield-fleet-key-v1"
BUS_ID = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
EPOCH = re.compile(r"^[0-9a-f]{1,32}$")


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


def _aad(bus_id, epoch, seq):
    return f"road-shield|v{VERSION}|{bus_id}|{epoch}|{int(seq)}".encode("utf-8")


def check_ids(bus_id, epoch):
    if not BUS_ID.match(str(bus_id)):
        raise PacketError("bus id must be 1-40 letters, digits, '.', '_' or '-'")
    if not EPOCH.match(str(epoch)):
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
