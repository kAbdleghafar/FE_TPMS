"""Export Abaqus results for comparison with tpmsfem.

Run with the Python that ships with Abaqus (not your normal Python):

    abaqus python abaqus_odb_export.py <job.odb> <out_prefix> [step_name]

Writes ``<out_prefix>_U.csv`` (node, u1, u2, u3) from the last frame of the
step (default: last step) and ``<out_prefix>_S.csv`` (elem, ip, s11, s22, s33,
s12, s13, s23) at the integration points. Then on your workstation:

    python -m tpmsfem.bench.compare ours.npz --abaqus <out_prefix>

Written for the odbAccess API (Abaqus 2017+, Python 2.7 and 3); it has not
been run by the tpmsfem authors -- please report issues.
"""
from __future__ import print_function

import sys

from odbAccess import openOdb  # noqa: F401  (Abaqus-only module)
from abaqusConstants import INTEGRATION_POINT, NODAL  # noqa: F401


def main(odb_path, prefix, step_name=None):
    odb = openOdb(odb_path, readOnly=True)
    steps = list(odb.steps.keys())
    step = odb.steps[step_name or steps[-1]]
    frame = step.frames[-1]
    print("step %s, frame %d (t=%g)" % (step.name, frame.frameId, frame.frameValue))
    inst = odb.rootAssembly.instances[list(odb.rootAssembly.instances.keys())[0]]

    U = frame.fieldOutputs["U"].getSubset(region=inst)
    with open(prefix + "_U.csv", "w") as f:
        f.write("node,u1,u2,u3\n")
        for v in U.values:
            d = v.data
            f.write("%d,%.12e,%.12e,%.12e\n" % (v.nodeLabel, d[0], d[1], d[2]))

    S = frame.fieldOutputs["S"].getSubset(region=inst, position=INTEGRATION_POINT)
    with open(prefix + "_S.csv", "w") as f:
        f.write("elem,ip,s11,s22,s33,s12,s13,s23\n")
        for v in S.values:
            d = v.data
            f.write("%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n"
                    % (v.elementLabel, v.integrationPoint, d[0], d[1], d[2], d[3], d[4], d[5]))
    odb.close()
    print("wrote %s_U.csv and %s_S.csv" % (prefix, prefix))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
