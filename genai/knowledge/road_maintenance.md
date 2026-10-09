# Road maintenance notes for the ROAD-SHIELD assistant

These notes are written for this project in plain words. They name the standards they draw on; for any
specification, contract or design decision, read the standard itself - these notes are a guide, not a substitute.

## Why potholes form

A pothole starts where water gets into the pavement. Rain enters through cracks or poorly sealed joints, weakens
the granular base and the bond between the bituminous layers, and every passing wheel then pumps fines out and
breaks the weakened surface. In India the monsoon is the main driver: a road that looked sound in May can be
badly potholed by August, because months of standing water and heavy traffic work on the same weak spots.
Poor drainage, a camber that does not shed water, overloaded trucks, thin or poorly compacted surfacing and
utility cuts that were never properly reinstated all make it worse.

Small cracks are therefore the cheapest point to intervene: sealing a crack costs a fraction of patching the
pothole it would become.

## Kinds of distress

ASTM D6433 (Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys) defines the distress
types surveyors record on asphalt roads, each at low, medium or high severity. The ones ROAD-SHIELD cares about:

- Alligator (fatigue) cracking: interconnected cracks forming many-sided pieces, from repeated wheel loads on a
  base that is too weak. It usually precedes potholes.
- Longitudinal and transverse cracking: single cracks along or across the road, from shrinkage, ageing of the
  binder or reflective cracking from layers below.
- Potholes: bowl-shaped holes in the surface, usually under about 750 mm across; severity depends on depth and
  diameter.
- Rutting: depressions in the wheel paths.
- Patching and utility-cut patching: earlier repairs, which are themselves counted as distress.

ROAD-SHIELD's classes map onto these: "Crack" covers longitudinal, transverse and alligator cracking; "Pothole
Cavity" is the pothole distress. Waterlogging, missing zebra crossings, missing dividers and damaged signs are
road-safety defects that the PCI does not include.

## Pavement Condition Index (PCI)

The PCI is a number from 0 (failed) to 100 (excellent) calculated from the types, severities and densities of
distress in a sample unit, using deduct curves. ASTM D6433 rates the result as Good (86-100), Satisfactory
(71-85), Fair (56-70), Poor (41-55), Very Poor (26-40), Serious (11-25) and Failed (0-10). ROAD-SHIELD uses
the same boundaries, labels its top band Excellent, and reports everything up to 25 as Failed.

ROAD-SHIELD's PCI is a per-frame proxy: it is computed from what one camera frame shows, not from a full
sample-unit survey, and it says so in every result. It is useful for ranking; a formal condition survey still
needs the standard procedure.

## Repairing a pothole

IRC:82 (Code of Practice for Maintenance of Bituminous Roads) is the Indian code for maintaining bituminous
roads, including patch repairs; follow it for specifications. The repair methods below are the common ones
described in the international literature (for example the US FHWA manual on materials and procedures for
pothole repair, FHWA-RD-99-168), from quickest to most durable:

- Throw-and-roll: cold or hot mix placed in the hole and compacted with the roller or a vehicle wheel.
  Fast and suited to emergencies and monsoon conditions, but short-lived.
- Cut-and-patch (semi-permanent): the damaged area is cut to a rectangle with vertical sides, loose material and
  water are removed, a tack coat is applied, and hot mix is laid in layers and compacted, then the edges are
  sealed. This is the normal planned repair.
- Spray injection: air blows out the hole, then binder and aggregate are sprayed in; quick with a dedicated rig.
- Full-depth repair: when the base has failed, the base is reconstructed before resurfacing; patching the
  surface alone over a failed base will fail again.

Good practice regardless of method: repair in dry conditions where possible, remove standing water, cut back
to sound material, apply tack coat, compact thoroughly in layers, finish level with the surrounding surface and
seal the joints so water cannot get in again.

For bituminous materials and laying, the MoRTH Specifications for Road and Bridge Works (Section 500) define
mixes such as Dense Bituminous Macadam (DBM) and Bituminous Concrete (BC). ROAD-SHIELD prices repairs in DBM
tonnage at an indicative rate; a tender must use the state PWD or CPWD Schedule of Rates.

## Work-zone safety

Any repair on a live road needs traffic management: advance warning signs, cones or barricades, a safe
working space and, where needed, flagmen. IRC:SP:55 (Guidelines on Traffic Management in Work Zones) sets out
layouts. Night work needs lighting and reflective devices.

## How urgent is a repair?

Urgency depends on how bad the defect is, how big it is and how much traffic crosses it. ROAD-SHIELD's
Priority Index combines exactly these three: PI = w1(100 - PCI) + w2 x volume + w3 x traffic, each term scaled
0-100, with bands P1 (repair within 24 hours) to P4. Deep potholes on busy roads come first; a hairline crack
on a quiet lane can wait for the next resurfacing cycle, though sealing it early is cheaper.

## Monsoon readiness

Before the monsoon: clear drains and culverts, seal cracks, fix potholes on high-traffic corridors first, and
stock cold mix for emergency repairs during the rains. During the monsoon, emergency throw-and-roll repairs
keep roads open; proper cut-and-patch repairs follow when the weather allows.
