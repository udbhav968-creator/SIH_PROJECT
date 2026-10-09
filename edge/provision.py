"""
Give one bus its own key.

    python -m edge.provision BUS-KA01-101            # prints the key for that bus
    python -m edge.provision BUS-KA01-101 --env      # as a line for the bus's environment file

Run it on the server (or wherever the fleet master key is kept), with ROAD_SHIELD_FLEET_KEY set to the master
key. Copy the printed key to that bus only, as its ROAD_SHIELD_FLEET_KEY. The bus seals packets with it; the
server derives the same key from the master key and the bus id the packet names, so nothing has to be stored
per bus, and a key taken from one bus cannot seal packets for any other bus.

To take a bus out of service (stolen, decommissioned): POST /api/v1/fleet/revoke {"bus_id": ...}. Once every
bus has been re-provisioned, set ROAD_SHIELD_ALLOW_FLEET_KEY=0 on the server so the master key itself is no
longer accepted from the road.

The key is printed to the terminal only. Do not paste it into chat, tickets or source control.
"""
import argparse
import sys

from edge import crypto


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("bus_id", help="bus id: 1-40 letters, digits, '.', '_' or '-'")
    ap.add_argument("--env", action="store_true", help="print as ROAD_SHIELD_FLEET_KEY=<key>")
    a = ap.parse_args(argv)
    master = crypto.load_key()
    if master is None:
        print("ROAD_SHIELD_FLEET_KEY (the fleet master key) is not set in this shell.", file=sys.stderr)
        return 2
    try:
        key = crypto.bus_key(master, a.bus_id).hex()
    except crypto.PacketError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"ROAD_SHIELD_FLEET_KEY={key}" if a.env else key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
