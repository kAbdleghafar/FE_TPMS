// Word report: process and microstructure results (TPMSsuite, Python package tpmsfem), November 2026.
// Usage: node tools/report/make_report.js docs/reports/<name>.docx  (figure sizes in img_sizes.json)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, PageBreak, Footer, PageNumber,
  TableOfContents, Header,
} = require("docx");

const REPO = path.resolve(__dirname, "../..");
const IMG = path.join(REPO, "docs/img");
const SIZES = JSON.parse(fs.readFileSync(path.join(__dirname, "img_sizes.json"), "utf8"));
const OUT = process.argv[2];

const FONT = "Arial";
const TEXT_W = 9638; // A4 with 2 cm margins, DXA
const INK = "1F1F1F", MUTED = "5A5A5A", ACCENT = "1F4E79", HEAD_FILL = "DCE6F0", ROW_FILL = "F4F6F8";

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold** markup inside a string
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter((p) => p.length);
  return parts.map((p) =>
    p.startsWith("**") ? new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, ...base })
                       : new TextRun({ text: p, font: FONT, ...base }));
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, font: FONT })] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text),
                                               spacing: { after: 60, line: 264 } });
const N = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: runs(text),
                                     spacing: { after: 60, line: 264 } });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = TEXT_W / total;
  const w = widths.map((x) => Math.round(x * scale));
  w[w.length - 1] += TEXT_W - w.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, header, shade, bold) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    borders,
    shading: header ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" }
                    : (shade ? { fill: ROW_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: i > 0 && opts.right ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(text), { size: 17, bold: header || bold, color: INK }),
    })],
  });
  const trs = [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true, false)) })];
  rows.forEach((r, k) => trs.push(new TableRow({
    children: r.map((c, i) => cell(c, i, false, k % 2 === 1, opts.boldRows && opts.boldRows.includes(k))),
  })));
  return [new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: w, rows: trs }),
          new Paragraph({ children: [], spacing: { after: 120 } })];
}

let figNo = 0;
function figure(file, caption, widthIn = 6.6) {
  figNo += 1;
  const [pw, ph] = SIZES[file];
  const wpx = Math.round(widthIn * 96);
  const hpx = Math.round(wpx * ph / pw);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(IMG, file)),
                                transformation: { width: wpx, height: hpx },
                                altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ spacing: { after: 200 }, children: [
      new TextRun({ text: `Figure ${figNo}. `, bold: true, font: FONT, size: 18, color: MUTED }),
      ...runs(caption, { size: 18, color: MUTED, italics: false })] }),
  ];
}
let tabNo = 0;
const tcap = (text) => { tabNo += 1; return new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: [
  new TextRun({ text: `Table ${tabNo}. `, bold: true, font: FONT, size: 18, color: MUTED }), ...runs(text, { size: 18, color: MUTED })] }); };

// ---------- content ----------
const body = [];

// Title page
body.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "TPMSsuite", font: FONT, size: 28, color: MUTED })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({
    text: "Process and microstructure results: melt-pool flow, porosity, solute trapping, 3-D cells with EDS maps, 3-D grain structure and stress-induced HCP",
    font: FONT, size: 44, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({
    text: "AM-Bench 2018 IN625 benchmarks and the CoCrFeMnNi graded-gyroid lattice", font: FONT, size: 26, color: INK })] }),
  new Paragraph({ spacing: { before: 600, after: 60 }, children: runs("Progress report, 19 November 2026 (revision 3: adds where twins or HCP could start in the compressed CoCrFeMnNi gyroid and how much HCP a metastable FeMnCoCr gyroid would form, Section 8; revision 2 added Section 6)", { size: 22, color: MUTED }) }),
  new Paragraph({ spacing: { after: 60 }, children: runs("Repository: kAbdleghafar/TPMSsuite (default branch main; Python package tpmsfem). Contributors: Khaled Abdelghafar and Claude (Anthropic).", { size: 20, color: MUTED }) }),
  new Paragraph({ spacing: { after: 60 }, children: runs("Every number in this report is produced by scripts in the repository (Appendix A); the detailed write-ups are in docs/freesurface_lbm.md, docs/porosity_maps.md and docs/process_models.md (§5.3–§5.6, §7.4); every calculation is explained in docs/manual.md.", { size: 20, color: MUTED }) }),
  pageBreak(),
);

// Contents
body.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "Contents", font: FONT })] }),
  ...[
    ["1. Summary", 0], ["2. What was built", 0], ["3. Melt-pool flow with a free surface", 0],
    ["3.1 Method · 3.2 Verification and case B · 3.3 Powder and calibration status", 1],
    ["4. Porosity process maps", 0], ["5. Cells and Nb segregation up the melt pool, with solute trapping", 0],
    ["6. Cells in 3-D with Nb, Mo, Cr and Fe, and what EDS would record", 0],
    ["6.1 Model and alloy data · 6.2 Run · 6.3 Cells and segregation · 6.4 Synthetic EDS maps · 6.5 Findings and limits", 1],
    ["7. Grain structure: 3-D cellular automaton", 0],
    ["7.1 Model and verification · 7.2 Single track against NIST's EBSD · 7.3 Ten-layer IN625 blocks · 7.4 CoCrFeMnNi gyroid over 20 layers", 1],
    ["8. Where twins or HCP could start: a CoCrFeMnNi gyroid in compression (Level 0)", 0],
    ["8.1 Criterion and calibration · 8.2 The gyroid column in compression · 8.3 Findings and limits · 8.4 From the as-built state · 8.5 How much HCP: a metastable FeMnCoCr gyroid (Level 1a)", 1],
    ["9. Problems found and fixed", 0], ["10. Limits and next steps", 0], ["Appendix A. Reproducing the results", 0],
  ].map(([t, lvl]) => new Paragraph({ indent: { left: lvl ? 400 : 0 }, spacing: { after: lvl ? 100 : 40 },
      children: [new TextRun({ text: t, font: FONT, size: lvl ? 18 : 22, color: lvl ? MUTED : INK, bold: !lvl })] })),
  new Paragraph({ spacing: { before: 240 }, children: runs("Figures: 1 flow solver (case B) · 2 porosity maps · 3 phase field up the pool · 4 3-D cells · 5 EDS maps · 6 CA vs EBSD · 7 IN625 blocks · 8 CoCrFeMnNi gyroid · 9 stacking faults in the compressed gyroid · 10 the as-built start · 11 HCP in a FeMnCoCr gyroid.", { size: 18, color: MUTED }) }),
  pageBreak(),
);

// 1 Summary
body.push(H1("1. Summary"));
body.push(P("This phase extended TPMSsuite (formerly tpmsfem) from conduction melt pools and rule-based grains to six new capabilities, each tested and checked against published data where data exist:"));
body.push(
  B("**Free-surface melt-pool flow solver** (our own implementation of the free-surface lattice Boltzmann method with an enthalpy heat solver, NumPy/CuPy). It is verified to round-off in mass and energy and predicts the AM-Bench case-B pool at 120 × 34.5 µm against the measured 123.5 × 36.0 µm. Calibrating it independently needs finer cells than a CPU affords."),
  B("**Porosity process maps** (lack of fusion and keyhole). For IN625 they reproduce 12 of 13 conditions reported by Locatelli et al. (2026). For the CoCrFeMnNi gyroid they place the study's 200 W / 0.75 m/s point on the lack-of-fusion limit near the build plate. A 70 µm hatch removes the risk."),
  B("**Phase field up the whole melt pool with solute trapping.** The cells refine from 0.57 µm at the fusion boundary to 0.28–0.31 µm higher up, inside the 0.2–1.0 µm measured by SEM, converged within 8 % in the interface width. Nb segregation weakens towards the top of the pool."),
  B("**3-D phase field with several elements, and synthetic EDS maps.** IN625 cells at the case-B mid-pool point, with Nb, Mo, Cr and Fe, form a columnar network whose edges carry Nb- and Mo-rich columns. The spacing of the material grown last is 0.33 µm (2-D: 0.34 µm; measured 0.2–1.0 µm) and was still coarsening. The segregation matches the 2-D model within 0.2 wt%. Simulated EDS maps show that SEM-EDS at 15 kV keeps only about a quarter of the Nb contrast of these cells, 5 kV keeps 85 % and STEM 96 %; Cr and Fe segregation stays below the counting noise."),
  B("**3-D grain cellular automaton.**"),
  B("On the AM-Bench case-B track it reproduces the measured grain count (90–99 against 92) and size (11–14 µm against 10.1 µm).", 1),
  B("In 10-layer IN625 blocks it develops a <100> fibre along the build direction: 35 % of the volume at 90° rotation and 68 % at 67° with 1 µm cells (random: 10 %).", 1),
  B("In the CoCrFeMnNi gyroid, columnar grains grow to about 100 µm in 20 layers.", 1),
  B("**Where twins or HCP could start under compression (Level 0 of the FCC → HCP question).** In a compressed CoCrFeMnNi gyroid, stacking faults (the precursor of both) can extend as soon as the walls yield, because the as-built yield stress (757 MPa) exceeds the measured twinning stress (720 MPa): 6 % of the lattice at 1 % strain and 13 % at 3 %, in the bending hinges of the walls. With a third lower fault energy (low-SFE variants, cryogenic loading) it is 28 % and 52 %; from the simulated as-built state the residual stresses bring this forward (6.5 times the fault-prone volume at 0.4 % strain). How much HCP forms (Level 1a): in a gyroid of the metastable Fe50Mn30Co10Cr10, an Olson–Cohen law fitted to LPBF EBSD data gives 2–3.5 % HCP on average at 5 % compression (0.94 % as built), 8–12 % in the bending hinges."),
);
body.push(P("**What limits the results now** is compute. Three things need a GPU:"));
body.push(
  B("a converged calibration of the flow model (5 µm cells or finer, with a clean interface);"),
  B("phase-field runs with D/V ≥ 10 W0 at the top of the pool (about 300 CPU-hours), and 3-D runs with a converged interface width, long enough for a steady spacing;"),
  B("textures on the lattice at ≤ 1 µm cells (about 700 million cells per window)."),
);
body.push(P("Several numerical problems were found and fixed along the way (Section 9); every result in this report was produced after the fixes."));

// 2 What was built
body.push(H1("2. What was built"));
body.push(tcap("New and extended code (all under the Apache-2.0 licence of the repository)."));
body.push(...table(["module / script", "purpose", "tests"], [
  ["process/freesurface.py, freesurface_kernels.py", "free-surface LBM melt flow (D3Q19 TRT, Körner free surface, Darcy damping) with enthalpy heat transport; Marangoni (constant or Sahoo), recoil, evaporation, radiation; fused numba lattice kernel", "tests/test_freesurface.py (8)"],
  ["process/powder.py", "powder bed: log-normal sizes, rain-and-roll packing, recoater, voxelisation", "tests/test_powder.py (2)"],
  ["process/defects.py; benchmarks/literature/porosity_maps.py", "lack-of-fusion criterion and unfused fraction, keyhole number; process maps", "tests/test_defects.py (3)"],
  ["process/phasefield.py", "solute trapping (continuous growth model) as velocity-corrected alloy parameters; adaptive stable time step (1.7× faster)", "tests/test_phasefield.py (11)"],
  ["benchmarks/literature/pf_path_trapping.py", "phase field along the case-B melt-pool path with checkpoint/resume", "—"],
  ["process/phasefield3d.py", "3-D quantitative phase field of a dilute multicomponent alloy: one supersaturation per solute, cubic anisotropy in any orientation, anti-trapping and trapping per solute, solidified 3-D composition stored, numba and NumPy/CuPy kernels, checkpoints, watershed cell counting", "tests/test_phasefield3d.py (12)"],
  ["post/eds.py", "synthetic SEM-EDS (X-ray generation volume against beam voltage and line) and STEM-EDS (thin foil, beam broadening) maps with counting noise and overlays", "tests/test_eds.py (6)"],
  ["benchmarks/literature/pf3d_in625_eds.py", "3-D IN625 cells at the case-B mid-pool point and their EDS maps", "—"],
  ["post/hcp_kinetics.py; benchmarks/literature/femncocr_gyroid_hcp.py", "HCP fraction against plastic strain (Olson–Cohen, exponential and sigmoidal shapes, one rate constant fitted to measured data; LPBF Fe50Mn30Co10Cr10 preset); the FeMnCoCr gyroid study", "tests/test_hcp_kinetics.py (3)"],
  ["post/fcc_faults.py; benchmarks/literature/cantor_gyroid_faults.py", "Shockley-partial bookkeeping, shear stress on the leading partial (twinning / transformation systems) for any stress and orientation, the free-partials bound; the compressed gyroid study", "tests/test_fcc_faults.py (8)"],
  ["process/phasefield3d.py, CUDA kernel (from a local GPU session)", "fused CUDA kernel for the 3-D phase field; the CuPy path passes the full test suite on an RTX 4070 SUPER", "tests/test_phasefield3d.py (on GPU)"],
  ["process/grain_ca.py", "3-D grain cellular automaton: decentred octahedra, KGT tip kinetics, nucleation, epitaxy, persistent state, scan-strategy events with rounded pool ends, optional heat-flow bias", "tests/test_grain_ca.py (6)"],
  ["nist_ambench2018_tracks_ca.py, ca_multilayer.py, cantor_gyroid_ca.py", "CA studies: case-B EBSD comparison, IN625 blocks, CoCrFeMnNi gyroid", "—"],
], [3.0, 5.2, 1.8]));

// 3 Free-surface flow
body.push(H1("3. Melt-pool flow with a free surface"));
body.push(P("**Why.** The conduction melt-pool model (calibrated on AM-Bench) has no fluid flow, no free surface and no powder. A colleague's high-fidelity LBM video motivated an equivalent capability. A feasibility check of waLBerla and OpenLB showed that neither code ships laser-melting physics, so the solver was written in TPMSsuite, following the published method of Lei et al. (2025)."));
body.push(H2("3.1 Method"));
body.push(
  B("**Flow:** D3Q19 lattice, two-relaxation-time collision with Guo forcing; Körner et al. (2005) free surface (gas, interface, liquid, wall cells; mass exchanged through streamed populations; mass conserved to round-off); Carman–Kozeny damping for melting and solidification."),
  B("**Heat:** enthalpy on the same cells, laser absorbed by Beer–Lambert along the metal. With energy_transport = \"lbm\" the enthalpy is carried with the lattice mass flux, so heat follows the moving surface exactly; the remaining residual is booked and reported."),
  B("**Surface physics:** Marangoni stress (constant dσ/dT or the Sahoo model), recoil pressure 0.54 p_sat(T), evaporation cooling, radiation."),
);
body.push(H2("3.2 Verification and case B"));
body.push(tcap("Verification checks."));
body.push(...table(["check", "result"], [
  ["mass conservation (droplets, collapsing column, melt pools)", "1e-13 relative"],
  ["Laplace pressure of a resting droplet", "within 1 % of 2σ/R"],
  ["surface shear stress on a periodic layer (Couette profile)", "within 1 % in every cell"],
  ["heat only, case B at 5 µm, against the conduction model", "0.3 % RMS temperature; pool 290 × 110 × 30 µm vs 295 × 110 × 30 µm"],
  ["energy with flow (carried with the mass)", "error equals the booked residual to 1e-10"],
], [6, 4]));
body.push(tcap("AM-Bench 2018-02 case B (179.2 W, 0.8 m/s, IN625), 10 µm cells."));
body.push(...table(["", "width", "depth", "peak T", "melt speed"], [
  ["measured (NIST AMMT)", "123.5 ± 6.5 µm", "36.0 ± 1.9 µm", "", ""],
  ["flow without Marangoni", "115 µm", "35.5 µm", "2450 °C", "0.02 m/s"],
  ["flow with Marangoni (dσ/dT = −1.1e-4 N/(m K))", "120 µm", "34.5 µm", "2320 °C", "0.3–0.35 m/s"],
], [4, 2, 2, 1.6, 1.8]));
body.push(...figure("freesurface_caseB.png", "Case B with the free-surface flow solver: temperature and flow in the melt pool, with and without Marangoni flow (10 µm cells)."));
body.push(H2("3.3 Powder and calibration status"));
body.push(
  B("**Powder vs plate** (5 µm cells, case-B laser, 40 µm powder layer at 0.44 packing): on the plate the track melts 158 × 43 µm (energy residual 0.1 %); on powder it melts 113 × 23 µm into the plate and builds a track 28 µm high (residual 5.4 %, qualitative)."),
  B("**Calibration in the flow model is not yet credible.** With Sahoo surface tension the depth lands on the measurement for sulfur activities 0.005–0.02, but at 10 µm the pool size is set by the lattice speed limiter, not the physics, and at 5 µm (4.7 h per run here) it is not converged (145 × 33 µm, limiter active in 742 cells)."),
  B("**Until a GPU is available,** the process studies use the conduction model's calibration (absorptivity 0.32, absorption depth 15 µm)."),
);

// 4 Porosity
body.push(H1("4. Porosity process maps"));
body.push(P("**Method.** For every laser power and scan speed, the calibrated conduction melt-pool model (with a powder layer) gives the pool width W and depth D. Lack of fusion follows the criterion of Tang et al. (2017), (h/W)² + (L/D)² ≤ 1, and a 2-D unfused fraction; keyholing follows the keyhole number Ke of Gan et al. (2021)."));
body.push(tcap("IN625 against Locatelli et al. (2026): 30 µm layers, 50 µm hatch, plate at 80 °C; predicted with a 30 µm spot radius."));
body.push(...table(["reported", "predicted"], [
  ["lack of fusion: 100 W at 1.0, 1.5, 2.0 m/s", "all three: criterion 1.37 → 1.77 → 2.18"],
  ["keyhole pores: 250 and 300 W at 0.5 m/s", "Ke 17.8 and 21.3, above the ~16 threshold; fully fused"],
  ["dense: 150–300 W at 1.0 and 1.5 m/s", "7 of 8 (the miss: 150 W at 1.5 m/s, criterion 1.35)"],
], [5, 5]));
body.push(tcap("CoCrFeMnNi gyroid point (200 W, 0.75 m/s, 30 µm layers) against the part temperature."));
body.push(...table(["part temperature", "W × D", "criterion, 85 µm hatch", "unfused", "criterion, 70 µm hatch", "Ke"], [
  ["80 °C (first layers)", "97 × 54 µm", "1.08", "0.7 %", "0.83", "9.0"],
  ["140 °C (~9 mm up)", "103 × 55 µm", "0.99", "0.2 %", "0.77", "9.5"],
  ["300 °C", "105 × 58 µm", "0.93", "0.03 %", "0.71", "11.0"],
  ["500 °C (upper bound)", "107 × 63 µm", "0.86", "0", "0.66", "13.6"],
], [2.6, 1.8, 1.8, 1.2, 1.8, 0.8]));
body.push(...figure("porosity_maps.png", "Porosity process maps. (a, b) IN625 with 30 and 40 µm spot radius against Locatelli et al. 2026; (c) CoCrFeMnNi with the gyroid study's point. Lack of fusion lies below the black line; keyholes above the orange lines."));
body.push(P("**Reading.** The maps separate the three regimes as reported. The 30 µm spot radius fits the IN625 machine better than 40 µm. The gyroid's lack-of-fusion risk sits in the first millimetres above the plate, where the part is coldest; a 70 µm hatch removes it everywhere, and keyholes stay below the pore threshold."));

// 5 Phase field
body.push(H1("5. Cells and Nb segregation up the melt pool, with solute trapping"));
body.push(P("**Set-up.** The 2-D quantitative phase field (Karma–Echebarria, IN625 as quasi-binary Ni–4 wt% Nb) runs at seven points up the centreline of the calibrated case-B pool, from the fusion boundary (R = 0.027 m/s) to the surface (R = 0.29 m/s)."));
body.push(
  B("**Solute trapping:** the continuous growth model as velocity-corrected alloy parameters, V_D = 1 m/s. The partition coefficient rises from 0.49 to 0.60 and the freezing range falls from 43 to 29 K."),
  B("**Resolution:** the interface width follows the diffusion length, W0 = min(10 nm, D/5R), down to 2.1 nm at the top. A convergence series (W0 = 10, 6.5, 4.1 nm) and one run without trapping check it."),
  B("**Cost:** about 6 h per top point on two cores, with an adaptive time step and checkpoints."),
);
body.push(tcap("Phase field up the case-B centreline (spacing from the counted cell boundaries)."));
body.push(...table(["height above fusion boundary", "R [m/s]", "k(V)", "W0 [nm]", "spacing [µm]", "c_s* cores [wt% Nb]", "c_l* tip liquid [wt%]", "95th pct. of solid [wt%]"], [
  ["0 µm", "0.027", "0.49", "10", "0.57", "3.27", "5.77", "8.1"],
  ["5 µm", "0.074", "0.52", "8.1", "0.37", "3.46", "5.52", "7.1"],
  ["10 µm", "0.092", "0.52", "10 / 6.5 / 4.1", "0.35 / 0.32 / 0.33", "3.47–3.50", "5.24–5.72", "6.2–7.2"],
  ["15 µm", "0.133", "0.54", "4.5", "0.28", "3.62", "5.51", "6.6"],
  ["20 µm", "0.173", "0.56", "3.5", "0.30", "3.72", "5.50", "6.0"],
  ["20 µm, no trapping", "0.173", "0.48", "3.5", "0.28", "3.60", "5.79", "6.7"],
  ["25 µm", "0.257", "0.59", "2.3", "0.30", "3.90", "5.56", "5.6"],
  ["30 µm", "0.287", "0.60", "2.1", "0.31", "3.90", "5.55", "6.0"],
], [2.2, 1.0, 0.9, 1.4, 1.5, 1.4, 1.4, 1.4]));
body.push(...figure("pf_path_trapping.png", "Phase field up the case-B pool with solute trapping: (a) growth rate and stability limits, (b) cell spacing against the measured range and the W0 convergence series, (c) Nb partitioning at the tips, (d, e) Nb near the tips at the fusion boundary and near the surface."));
body.push(P("**Findings.**"));
body.push(
  B("**The cells refine from the fusion boundary, then level off.** The spacing falls from 0.57 µm to 0.37 µm 5 µm above the boundary and stays at 0.28–0.31 µm up to the surface. This is inside the 0.2–1.0 µm measured by SEM (Ghosh et al. 2018) and their own phase-field range (0.20–0.43 µm). Hunt's law trends the other way."),
  B("**The resolution is adequate.** W0 = 10, 6.5 and 4.1 nm give 0.35, 0.32 and 0.33 µm at 10 µm height: within one window step. The earlier concern that the W0 = 10 nm points were 30 % too low is not confirmed."),
  B("**Segregation weakens up the pool.** Cell cores go from 3.27 to 3.90 wt% Nb (alloy 4 wt%); the effective partition from 0.57 to 0.70; the intercellular 95th percentile from 8.1 to about 6 wt%. Less Laves phase is therefore expected in the upper part of each pool."),
  B("**Trapping contributes.** At 20 µm, switching it off lowers the cores from 3.72 to 3.60 wt%, raises the tip liquid from 5.50 to 5.79 wt% and the tip undercooling from 32 to 41 K. At the fusion boundary it changes nothing."),
  B("**No planar front inside the pool:** with trapping, absolute stability falls to 0.40 m/s at the top, where R = 0.29 m/s; the cells there are shallow but remain cells."),
);
body.push(P("**Limits.** V_D = 1 m/s is assumed (0.5–2 m/s are used for Ni–Nb); kinetic undercooling and solute drag are not included; the model is 2-D with a frozen temperature field and a binary alloy."));

// 6 3-D multicomponent phase field and EDS
body.push(H1("6. Cells in 3-D with Nb, Mo, Cr and Fe, and what EDS would record"));
body.push(P("**Question.** What do the cells of the case-B melt pool look like in 3-D when all the main alloying elements segregate, and how much of that segregation can an EDS map show? The 2-D, one-solute phase field of Section 5 cannot answer either: 2-D spacings are larger than 3-D ones, and an EDS map shows several elements at the resolution of the instrument."));
body.push(H2("6.1 Model and alloy data"));
body.push(
  B("**Phase field** (process/phasefield3d.py): the model of Section 5 in 3-D with one supersaturation per solute. The interface temperature sees their sum, each weighted by the solute's share of the freezing range. Every solute has its own anti-trapping current and solute trapping (continuous growth model, V_D = 1 m/s). The anisotropy is cubic, with <100> along the build direction."),
  B("**Verification:** with one solute and a box one cell thick the model reproduces the 2-D model to 1e-15; splitting a solute into two identical halves changes nothing; the numba and NumPy/CuPy kernels agree; every solute is conserved to round-off."),
);
body.push(tcap("IN625 of the AM-Bench 2018 feedstock as a dilute four-solute alloy."));
body.push(...table(["solute", "content [wt%]", "k (equilibrium)", "k at 0.092 m/s", "liquidus slope [K/wt%]", "share of the freezing range"], [
  ["Nb", "3.75", "0.48", "0.52", "−10.5 (Ni–Nb, as in Section 5)", "90 %"],
  ["Mo", "8.83", "0.85", "0.86", "−2.9 (Ni–Mo liquidus secant, estimate)", "10 %"],
  ["Cr", "20.7", "1.02", "1.02", "partitions only", "0"],
  ["Fe", "0.72", "1.09", "1.08", "partitions only", "0"],
], [1.0, 1.4, 1.5, 1.5, 3.2, 1.6]));
body.push(
  B("**Sources:** composition by ICP from Keller et al. (2017, Table 1); partition coefficients from Gan et al. (2019, Table 4, after Hu et al. 2017). The freezing range is 47 K at equilibrium and 40 K with trapping. One liquid diffusivity, 3e-9 m²/s."),
  B("**Ni** is the balance of the four solutes; the ~1 wt% of Ti, Al, Co and Si is not modelled. Cr and Fe have k close to 1: their effect on the interface temperature is below 2 % of the freezing range, so they are carried without it."),
  B("**EDS** (post/eds.py). SEM: the X-ray generation volume reaches the Anderson–Hasler range of each line, widens with depth as depth^1.5 and holds 90 % of the signal within that range. STEM: a 100 nm foil at 200 kV, the average through the foil plus about 5 nm of beam broadening. Both with Poisson counting noise and a continuum background; concentration to counts is linear (no ZAF corrections)."),
);
body.push(H2("6.2 Run"));
body.push(P("The mid-pool point of Section 5, 22.5 µm below the surface (G = 6.6e6 K/m, R = 0.092 m/s): a periodic 1.6 × 1.6 µm box with a 1.2 µm window, grown 5 µm, W0 = 10 nm (200 × 200 × 150 cells; the 2-D spacing at this W0 is within 8 % of the converged value). 60,000 steps took 2.2 h on 4 CPU cores. The 2-D reference uses the same multicomponent model, one cell thick and 2.4 µm wide."));
body.push(...figure("pf3d_in625_eds_3d.png", "3-D IN625 cells at the case-B mid-pool point: (a) the cell front at the end of the run, coloured by height; (b) Nb on three faces of the solidified block; (c) the Nb-rich columns along the cell edges (iso-surface)."));
body.push(H2("6.3 Cells and segregation"));
body.push(tcap("2-D and 3-D, same model; compositions over the material grown after the first third of the run."));
body.push(...table(["", "2-D", "3-D"], [
  ["cell spacing, material grown last (0.6–1 µm below the tips)", "0.34 µm", "0.32–0.33 µm"],
  ["cell spacing, material grown 4 / 5 µm earlier", "", "0.27 / 0.24 µm"],
  ["Nb: 5 % / median / 95 % [wt%]", "3.15 / 3.37 / 5.95", "2.95 / 3.44 / 5.84"],
  ["Mo: 5 % / median / 95 % [wt%]", "8.54 / 8.66 / 9.79", "8.42 / 8.67 / 9.77"],
  ["Cr: 5 % / median / 95 % [wt%]", "20.44 / 20.75 / 20.78", "20.45 / 20.75 / 20.82"],
  ["Fe: 5 % / median / 95 % [wt%]", "0.68 / 0.73 / 0.73", "0.68 / 0.73 / 0.74"],
  ["Ni (balance): 5 % / median / 95 % [wt%]", "63.1 / 66.5 / 66.8", "63.3 / 66.4 / 67.1"],
], [5.2, 2.4, 2.4]));
body.push(P("The 3-D spacing is sqrt(A/N) on transverse sections, with cells counted by watershed; for hexagonal cells the centre-to-centre distance is 7 % larger. Case-B cells measured by SEM: 0.2–1.0 µm (Ghosh et al. 2018)."));
body.push(H2("6.4 Synthetic EDS maps"));
body.push(tcap("Transverse section 0.8 µm below the tips: Nb contrast (5–95 % spread) kept in a noise-free map."));
body.push(...table(["technique", "Nb, Mo line", "90 % of the signal within", "Nb contrast kept"], [
  ["SEM, 15 kV", "Lα", "0.68 µm (Ni Kα: 0.45 µm)", "27 %"],
  ["SEM, 5 kV", "Lα (also Cr, Fe, Ni Lα)", "0.08 µm (Ni Lα: 0.11 µm)", "85 %"],
  ["STEM, 100 nm foil, 200 kV", "Kα", "0.01 µm", "96 %"],
], [2.6, 2.6, 3.0, 1.8]));
body.push(P("Figure 5, at the end of this section, shows the simulated section and the maps each technique would record, with profiles across the cells and longitudinal sections."));
body.push(H2("6.5 Findings and limits"));
body.push(
  B("**In 3-D the cells form a network of hexagon-like columns.** The front is a field of rounded cell caps over narrow, deep liquid grooves. Nb and Mo pile up along the cell walls and most of all along the cell edges, where three cells meet; these triple junctions form Nb-rich columns (up to 7–8 wt% Nb in the solid section). Cr, Fe and Ni are slightly depleted there. Isolated droplets of the last liquid reach 13–21 wt% Nb: the sites where Laves phase or carbides form in the real alloy (not modelled)."),
  B("**The 3-D spacing was still coarsening when the run ended.** The front first breaks up into fine cells, then cells are eliminated: 0.24 µm in the material grown 5 µm before the end, 0.33 µm in the material grown last. The tips were still drifting ahead of the T0 isotherm (1.2 µm mid-run, 1.4 µm at the end). 0.33 µm is therefore a lower bound for the steady 3-D spacing; it is close to the 2-D value and inside the measured range."),
  B("**Segregation agrees between 2-D and 3-D.** The 5–95 % ranges of Nb, Mo, Cr and Fe agree within 0.2 wt%; the 3-D cores are about 0.2 wt% leaner in Nb. The cheaper 2-D model gives the segregation statistics; 3-D is needed for the shape of the network (edges against walls, droplets, sections at any angle)."),
  B("**Only Nb and Mo segregation is measurable by EDS.** Cr varies by 2 % of its content and Fe by 9 % of its 0.7 wt%; both stay below the counting noise of a map in every technique (Fe because its signal is only about 11 counts per pixel). Ni depletion at the walls (about 5 %) is visible in STEM."),
  B("**SEM-EDS at 15 kV cannot resolve these cells.** At 15 kV, 90 % of the Nb and Mo signal comes from a volume 0.7 µm wide, two cells across, and the map keeps about a quarter of the Nb contrast. At 5 kV (L lines for all elements) the volume shrinks below 0.1 µm and 85 % survives; STEM keeps nearly all of it. SEM-EDS at the usual 15–20 kV therefore underestimates intercellular segregation in LPBF alloys several-fold: quantitative comparison needs STEM-EDS, low-kV SEM, or the same blurring applied to the simulation first."),
  B("**Uncertain inputs.** The SEM numbers depend on the shape assumed for the generation volume: a single Gaussian with the same 90 % diameter would keep under 1 % of the contrast at 15 kV, against 27 % here; a measured line scan would calibrate it. The Mo liquidus slope is an estimate (Mo carries 10 % of the freezing range); one liquid diffusivity is used for all solutes. A steady and converged 3-D spacing needs longer runs below W0 = 10 nm (GPU)."),
);
body.push(...figure("pf3d_in625_eds.png", "Simulated composition of a transverse section 0.8 µm below the tips (top row) and the maps an EDS would record with counting noise: SEM at 15 kV and 5 kV, STEM on a 100 nm foil; overlays of Nb (red), Cr (blue) and Ni (green). Bottom: Nb and Mo across the cells, noise-free, and longitudinal sections (simulated, SEM at 15 kV, STEM).", 6.4));

// 7 CA
body.push(H1("7. Grain structure: 3-D cellular automaton"));
body.push(H2("7.1 Model and verification"));
body.push(
  B("**Model** (Gandin & Rappaz 1997, as in ExaCA): each solid cell next to the liquid carries an octahedral envelope aligned with its grain's <100> axes, growing at the KGT dendrite-tip velocity; captured cells receive decentred envelopes, so growth follows the crystal, not the grid. Nucleation is optional; unmelted cells seed the pool (epitaxy)."),
  B("**Thermal input:** the steady thermal history of one track is shifted along every scan vector, with rounded pool ends where the laser switches on or off. Remelting and regrowth across tracks and layers are included; preheating between tracks is not."),
  B("**Heat-flow bias α** (optional): the growth axis turns towards the heat flow; α = 0 is dendritic, α = 1 the cellular limit."),
  B("**Verification:** a tilted grain grows as an octahedron along its own axes (extent ratio 0.870 vs exact 0.866 at 30°, 0.707 vs 0.707 at 45°); directional growth from 120 random grains selects aligned ones (misalignment 32° → 10°); KGT gives 31 K at 0.1 m/s, matching the phase-field tips (28–40 K)."),
);
body.push(H2("7.2 Single track against NIST's EBSD section (case B)"));
body.push(tcap("CA regrowth of the case-B pool, 0.5 µm cells, compared pixel by pixel with the measured section."));
body.push(...table(["CA run", "pixels within 10° of measured", "grains (measured 92)", "area-weighted size (measured 10.1 µm)"], [
  ["measured substrate extended along the scan", "25.4 %", "47", "15.3 µm"],
  ["same, other scan direction", "21.7 %", "49", "16.6 µm"],
  ["same, 0.25 µm cells", "26.8 %", "47", "16.3 µm"],
  ["same, cellular growth (α = 1)", "23.8 %", "51", "13.8 µm"],
  ["same, nucleation 1e15 /m³ at 30 K (20 % of pool nucleated)", "24.1 %", "60", "14.1 µm"],
  ["**stand-in 3-D substrate, 7.9 µm grains (3 runs)**", "12–17 %", "**90–99**", "**11.2–14.0 µm**"],
  ["rule-based gradient trace (previous work)", "27.5 %", "83", "12.3 µm"],
], [4.4, 2.0, 1.6, 2.4]));
body.push(...figure("ca_ambench_caseB.png", "3-D grain CA against NIST's EBSD section of the case-B track: measured and predicted sections (IPF along the build direction), a longitudinal section and a plan view, and the point-wise agreement of all models."));
body.push(P("**Reading.** With a statistically equivalent substrate the CA reproduces the grain count and size. Point by point it matches the simpler rules (about 25 %), independently of cell size, kinetics and growth mode: the limit is the substrate that NIST measured in one plane only."));
body.push(H2("7.3 Ten-layer IN625 blocks: texture and grain width"));
body.push(tcap("IN625 blocks at the AM-Bench settings (100 µm hatch, 20 µm layers), layer 10, 480 µm blocks unless stated."));
body.push(...table(["run", "<100> within 15° of BD", "grain width", "grains"], [
  ["**90°, 1 µm cells**", "**35 %**", "56 µm", "237"],
  ["**67°, 1 µm cells**", "**68 %**", "72 µm", "164"],
  ["90°, 1.5 µm cells", "17 %", "56 µm", "265"],
  ["90°, 2 µm cells, three substrate realisations", "15 / 14 / 24 %", "57 / 60 / 60 µm", "230 / 236 / 220"],
  ["67°, 2 µm cells", "26 %", "55 µm", "193"],
  ["90° / 67°, 1 µm cells, 240 µm block", "19 / 21 %", "55 / 48 µm", "79 / 56"],
  ["random orientations", "10 %", "", ""],
], [4.2, 2.2, 2.2, 1.6]));
body.push(...figure("ca_multilayer.png", "Ten-layer IN625 blocks: vertical sections and plan views (IPF along BD) at 90° and 67° rotation, the <100> fibre and grain width against layer for all runs, and {100} pole figures."));
body.push(
  B("**A <100> fibre builds up layer by layer, and 67° makes it stronger than 90°** in every comparable pair of runs."),
  B("**The fibre needs fine cells:** at 2 µm it stalls at 14–26 %, well beyond the ±5-point scatter between substrate realisations. Whether 1 µm is converged needs a GPU check at 0.5 µm."),
  B("**It also needs a large block:** 240 µm blocks give only 19–21 %, most likely because many of their few remaining grains touch the side walls."),
  B("**Grain width** grows from 15 µm to 55–60 µm in ten layers wherever the fibre is weak; with the strong 67° fibre the grains reach 72 µm."),
);
body.push(H2("7.4 CoCrFeMnNi gyroid over 20 layers"));
body.push(P("The CA builds 20 layers (0.7–1.3 mm build height) of the graded gyroid in two 1 × 1 mm windows, thin walls and thick walls, at the study's settings (200 W, 0.75 m/s, 30 µm layers, 85 µm hatch, contour plus hatch, no rotation). The cells are 2.5 µm (43 million per window)."));
body.push(tcap("Gyroid windows: area-weighted grain width and <100> fraction along BD."));
body.push(...table(["", "layer 1", "layer 5", "layer 10", "layer 20"], [
  ["thin walls: grain width", "28 µm", "47 µm", "64 µm", "98 µm"],
  ["thin walls: <100> within 15° of BD", "10 %", "16 %", "18 %", "13 %"],
  ["thick walls: grain width", "29 µm", "49 µm", "70 µm", "103 µm"],
  ["thick walls: <100> within 15° of BD", "8 %", "11 %", "11 %", "8 %"],
  ["thin walls, cellular growth α = 1: grain width", "", "38 µm", "53 µm", "84 µm"],
], [4.0, 1.5, 1.5, 1.5, 1.5]));
body.push(...figure("cantor_gyroid_ca.png", "CoCrFeMnNi graded gyroid, 20 layers: plan views and vertical sections of thin and thick walls (IPF along BD), the <100> fibre and grain width against layer (dotted: cellular growth), and {100} pole figures."));
body.push(
  B("**Grains coarsen steadily** in both thin and thick walls, from 28 µm to about 100 µm in 20 layers, and have not levelled off. Without rotation the grains are elongated along the hatch direction, with boundaries on the 85 µm track spacing."),
  B("**Down-facing walls carry dross:** each layer's pool reaches into the powder below the overhang, leaving a step of stray grains per layer."),
  B("**The growth mode matters over many layers:** cellular growth (α = 1) leaves 100 grains instead of 69 in the thin walls at layer 20, 15 % narrower."),
  B("**The textures are lower bounds:** at 2.5 µm cells the fibre stays weak (up to 19 % in thin walls, random in thick ones), and the IN625 blocks show that coarse cells blunt it. Whether the CA reproduces the strong <001> fibre reported for LPBF CoCrFeMnNi needs 1 µm cells on the lattice."),
);

// 8 Stacking faults under compression (Level 0)
body.push(H1("8. Where twins or HCP could start: a CoCrFeMnNi gyroid in compression (Level 0)"));
body.push(P("**Question.** If the CoCrFeMnNi gyroid is compressed, can a stress-induced FCC → HCP transformation occur, and where? Twins and HCP both start from stacking faults that extend; Level 0 maps where, and at what applied strain, the local stress can extend them. Which of the two then forms is set by the FCC/HCP free-energy difference, the subject of Level 1."));
body.push(H2("8.1 Criterion and calibration"));
body.push(
  B("**Criterion** (post/fcc_faults.py). For every element stress and grain orientation: the largest shear stress on the 12 {111}<112> systems of the leading Shockley partial (the twinning / transformation systems of crystal-plasticity models). A fault extends where it reaches τ_c. Each grain takes the stress of its element (a Sachs-type assumption), and only yielded material counts (nucleation needs dislocation sources)."),
  B("**Calibration.** τ_c = 360 MPa: the best-oriented grains (Schmid factor 0.5) reach it at the measured twinning stress of CoCrFeMnNi, 720 MPa at 77 and 293 K (Laplanche et al. 2016). Lower thresholds stand for lower effective fault energies, in proportion."),
  B("**A stricter bound** lets both partials move freely (Byun 2003): with the measured fault energy, 30 mJ/m² (Okamoto et al. 2016), it needs about twice the measured twinning stress."),
  B("**Checks.** The Schmid factors reproduce the twinning asymmetry of FCC single crystals ([001]: 0.47 in compression, 0.24 in tension; [111]: 0.31 and 0.16), and the partials reproduce the stacking shift between {111} layers (8 tests)."),
);
body.push(H2("8.2 The gyroid column in compression"));
body.push(P("The graded gyroid column of the study (4 × 4 × 8 mm, walls 0.2 → 0.6 mm), 0.1 mm voxels, relative density 0.29, compressed along the build direction between bonded platens with J2 plasticity (E = 203 GPa, as-built yield 757 MPa, hardening 1500 MPa assumed) and the grain-CA textures of the thin and thick walls; no residual stress."));
body.push(tcap("Share of the lattice volume where faults can extend (yielded material, CA textures)."));
body.push(...table(["nominal compressive strain", "0.4 %", "0.6 %", "1 %", "2 %", "3 %"], [
  ["nominal stress [MPa]", "59", "70", "78", "85", "88"],
  ["yielded volume", "8 %", "21 %", "39 %", "59 %", "69 %"],
  ["**τ_c = 360 MPa (CoCrFeMnNi twinning onset)**", "**1.0 %**", "**3.1 %**", "**6.2 %**", "**10.4 %**", "**13.1 %**"],
  ["τ_c = 300 MPa (fault energy × 0.83)", "3.1 %", "8.7 %", "16.6 %", "26.5 %", "31.9 %"],
  ["τ_c = 240 MPa (× 0.67)", "5.4 %", "14.8 %", "27.8 %", "43.6 %", "51.7 %"],
  ["τ_c = 120 MPa (× 0.33)", "8.2 %", "21.4 %", "38.6 %", "58.7 %", "68.5 %"],
  ["free-partials bound, 30 mJ/m²", "0", "0", "0", "0", "0"],
], [3.6, 1.2, 1.2, 1.2, 1.2, 1.2]));
body.push(...figure("cantor_gyroid_faults.png", "Stacking-fault extension in the compressed CoCrFeMnNi gyroid column: (a) nominal stress and yielded volume; (b) lattice volume where faults can extend against strain for five thresholds (solid: CA texture; dashed: random); (c) leading-partial Schmid factors of the textures in compression along the build direction; (d–f) fraction of grains that can extend faults on an x–z section; (g) plastic strain."));
body.push(H2("8.3 Findings and limits"));
body.push(
  B("**Faults can extend as soon as the walls yield.** The as-built yield stress is already above the measured twinning stress, so grains with a leading-partial Schmid factor ≥ 0.48 reach τ_c as their wall yields; hardening brings in more. The fault-prone material follows the plastic zones, the bending hinges of the inclined walls (plastic strain up to about 7 % at 3 % nominal strain)."),
  B("**HCP or twins.** The equiatomic alloy forms HCP only at hydrostatic pressures of order 10–20 GPa (Tracy et al. 2017; Zhang et al. 2017), so at room temperature these faults would most likely become twins. With a third lower effective fault energy (lower-SFE variants, cryogenic loading, thick HCP embryos) the fault-prone volume is four times larger; below a third of the room-temperature value every yielded region qualifies, where a stress-induced FCC → HCP transformation becomes plausible."),
  B("**Texture and wall thickness hardly matter:** the walls are inclined and each is loaded along its own direction, so the CA and random textures agree within 0.1 point, and the thin and thick halves within 0.2 point."),
  B("**Resolution.** A 0.05 mm run (stopped after 2 % strain by a container restart) gives the stress within 2 %, 8–14 % more fault-prone volume at τ_c = 360 MPa and within 4–9 % at 240 MPa; its free-partials maximum reaches 28.7 mJ/m² against 22.5, so thin-wall stress peaks are under-resolved at 0.1 mm."),
  B("**Limits.** Grain-to-grain stresses are not resolved (crystal plasticity, Level 2); the hardening is assumed; the threshold is calibrated on wrought, coarse-grained CoCrFeMnNi; room temperature, compression along the build direction, a column two cells tall. The as-built start follows in 8.4."),
);

body.push(H2("8.4 From the as-built state"));
body.push(P("The thermo-mechanical build of the column (layer lumps on a 1 mm plate, cooling to 20 °C, release onto a 3-2-1 support) is re-run and its state kept: residual stress and build plastic strain. The column is then compressed (plate bottom held, lattice top moved down) and compared with a stress-free model on the same mesh, plate and supports. Only material that yields during the compression counts. Before any load, the residual stress alone (median von Mises 584 MPa) already gives τ* ≥ 360 MPa in 3.7 % of the lattice."));
body.push(tcap("Compression from the simulated as-built state against the stress-free column (as-built / stress-free)."));
body.push(...table(["nominal compressive strain", "0.4 %", "1 %", "2 %", "3 %"], [
  ["nominal stress [MPa]", "44 / 58", "71 / 78", "84 / 85", "88 / 88"],
  ["yielding during compression", "39 / 7.5 %", "60 / 38 %", "75 / 59 %", "82 / 69 %"],
  ["**faults can extend, τ_c = 360 MPa**", "**5.5 / 0.8 %**", "**9.4 / 6.1 %**", "**13.6 / 10.4 %**", "**16.2 / 13.0 %**"],
  ["faults can extend, τ_c = 240 MPa", "27 / 4.9 %", "42 / 28 %", "55 / 44 %", "61 / 52 %"],
], [3.6, 1.6, 1.6, 1.6, 1.6]));
body.push(...figure("cantor_gyroid_faults_asbuilt.png", "The gyroid column compressed from its simulated as-built state against the same column stress-free: (a) nominal stress; (b) lattice volume where faults can extend for three thresholds; (c, d) fraction of grains that can extend faults (τ_c = 360 MPa) on an x–z section at 1 % strain."));
body.push(B("**The residual stresses bring the onset forward:** where they add to the load the walls yield earlier, giving 6.5 times the fault-prone volume at 0.4 % strain; the difference shrinks to 1.5 times at 1 % and 1.25 times at 3 %, and the stress–strain curves meet at about 2.5 %. In as-built lattices the first twins should therefore appear at smaller strains than a stress-free model predicts; a stress-relief anneal would remove most of the difference."));
body.push(H2("8.5 How much HCP: a metastable FeMnCoCr gyroid (Level 1a)"));
body.push(P("The equiatomic CoCrFeMnNi twins rather than transforming at room temperature, so its HCP fraction stays at zero. To show how much HCP a transforming alloy would form, the same gyroid column is compressed (to 5 %) with the metastable Fe50Mn30Co10Cr10, for which LPBF data exist (Micromachines 15 (2024) 123): yield 487.6 MPa, tensile strength 744.9 MPa, elongation 22.3 %, HCP 0.94 % as built and 28.1 % at 20 % strain (EBSD, tension)."));
body.push(
  B("**Kinetics** (post/hcp_kinetics.py): an Olson–Cohen law in each element's equivalent plastic strain, f = f0 + (1 − f0)[1 − exp(−β(1 − exp(−α ε_p))^n)], fitted to the measured point. One point fixes one rate constant; an exponential (n = 1) and a sigmoidal (n = 2, α = 5) shape bound the rest."),
  B("**Coupling** is one-way: the measured flow curve already contains the TRIP hardening (linear hardening 1300 MPa estimated from the tensile strength; E = 200 GPa and ν = 0.28 assumed)."),
);
body.push(tcap("HCP in the compressed Fe50Mn30Co10Cr10 gyroid column (exponential / sigmoidal kinetics; 0.94 % as built)."));
body.push(...table(["nominal compressive strain", "1 %", "2 %", "3 %", "5 %"], [
  ["nominal stress [MPa]", "53", "56", "58", "62"],
  ["yielded volume", "51 %", "68 %", "76 %", "85 %"],
  ["plastic strain, mean / 95th percentile", "0.2 / 1.1 %", "0.6 / 2.6 %", "0.9 / 4.1 %", "1.7 / 7.0 %"],
  ["**HCP, lattice mean**", "**1.3 / 1.0 %**", "**1.8 / 1.2 %**", "**2.4 / 1.4 %**", "**3.5 / 2.2 %**"],
  ["HCP, 95th percentile", "2.7 / 1.2 %", "5.1 / 2.2 %", "7.4 / 3.7 %", "11.6 / 7.7 %"],
  ["volume with ≥ 10 % HCP", "0 / 0", "0.4 / 0.1 %", "2.0 / 0.5 %", "7.1 / 3.2 %"],
], [3.6, 1.6, 1.6, 1.6, 1.6]));
body.push(...figure("femncocr_gyroid_hcp.png", "Strain-induced HCP in a gyroid of the metastable Fe50Mn30Co10Cr10: (a) the two fitted kinetic laws against the measured fractions; (b) lattice mean and 95th percentile of the HCP fraction against compressive strain; (c, d) HCP on an x–z section at 5 % strain for both laws; (e) plastic strain."));
body.push(
  B("**Little HCP at typical lattice-test strains, concentrated in the hinges:** the lattice-average HCP rises only to 2–3.5 % at 5 % compression, because the plastic strain is very uneven (lattice mean 1.7 %), while the bending hinges reach 8–12 %. A diffraction measurement averaged over the lattice would see little; one focused on the hinges, or EBSD of an interrupted sample, would see the transformation."),
  B("**The small-strain shape of the kinetics is the main uncertainty** (a factor 1.6–2.3 between the shapes); an HCP fraction measured at 2–10 % strain would fix it. The stress-state dependence (Stringfellow et al. 1992) and the transformation strain are not included (Level 1b)."),
);

// 9 Problems fixed
body.push(H1("9. Problems found and fixed"));
body.push(P("All results in this report were produced after these fixes; affected runs were repeated."));
body.push(
  N("**Free surface:** walls seen as gas in the curvature (fixed with mirror walls); half the surface area in the surface forces (fixed with gas-face areas); energy created at the moving surface (fixed by carrying enthalpy with the mass); powder runs blowing up (solid–solid mass exchange, laser absorption, recoil cap and speed limiter)."),
  N("**Porosity maps:** depth measured below the plate instead of from the powder top; a key mismatch between stored and looked-up results."),
  N("**Phase field:** a 30-chunk run could not survive container restarts; jobs now checkpoint after every chunk and resume with identical results."),
  N("**Grain CA:**"),
);
body.push(
  B("envelopes kept growing while their liquid neighbours were above the liquidus, which stalled the time step (growth now waits, and the envelope size is capped);", 1),
  B("the top 2 µm of the pool was never melted, which leaked measured orientations into the comparison and inflated the first agreement figure from 25 % to 31 %;", 1),
  B("square pool ends left 10 % of thin gyroid walls unmelted; rounding every segment end then remelted cells along contours (now rounded only where the laser switches on or off);", 1),
  B("diagonal tracks ran out of memory at 1 µm cells (candidates now built in chunks; events identical).", 1),
);
body.push(N("**3-D phase field, cell counting:** a global (Otsu) threshold merged all cell cores into one when the walls are weak next to strongly enriched triple junctions, reporting one cell per section; cells are now counted by a periodic watershed from the composition minima (new test with weak walls)."));
body.push(N("**EDS, SEM generation volume:** the first model (one Gaussian of the generation range) kept under 1 % of the Nb contrast at 15 kV. It was replaced by a depth-resolved volume, cusped as Monte Carlo volumes are and calibrated on the same 90 % diameter; the sensitivity is reported in Section 6.5."));
body.push(N("**Withdrawn claims:** an explanation of the CA's point-wise agreement (too-strong selection) that a later test disproved, and an unsupported statement about the rotation dependence of published textures."));

// 10 Next steps
body.push(H1("10. Limits and next steps"));
body.push(tcap("Open items, in order of value."));
body.push(...table(["item", "needs", "what it would settle"], [
  ["calibrate the flow model on AM-Bench cases A–C", "GPU; ≤ 5 µm cells with a clean interface", "independent absorptivity; Marangoni model; pool shape over powder"],
  ["CA textures at ≤ 1 µm (0.5 µm check) and with periodic side walls", "GPU; periodic boundaries (code change)", "a converged <100> fibre; the lattice texture against reported LPBF CoCrFeMnNi"],
  ["calibrate the heat-flow bias α", "measured grain widths or textures of the lattice", "growth mode for many-layer predictions"],
  ["phase field with D/V ≥ 10 W0 at the top of the pool; 3-D runs at converged W0 and long enough for a steady spacing", "GPU (~300 CPU-hours otherwise)", "final spacing and segregation near the surface; the steady 3-D spacing"],
  ["calibrate the SEM generation volume", "a measured EDS line scan across a known feature", "absolute SEM-EDS contrast (Section 6.4)"],
  ["Mo (and minor-element) liquidus slopes from CALPHAD", "a thermodynamic database", "the Mo share of the freezing range"],
  ["preheating and part-scale heat in the CA thermal input", "couple the part-scale thermal model", "grain growth in hot, thin walls"],
  ["measured textures / EBSD of the gyroid builds", "experiment", "validation of §7.4"],
  ["Level 1b: transformation strain and stress-state-dependent HCP kinetics in the constitutive update", "HCP fractions measured at several strains and stress states", "the small-strain kinetics (Section 8.4) and the TRIP effect on the lattice response"],
  ["in-situ diffraction during compression of a printed gyroid", "synchrotron or neutron beamtime", "HCP peaks and phase fractions against strain: validation of Section 8"],
], [3.6, 3.0, 3.4]));
body.push(P("**Compute note.** This cloud machine is reclaimed when the session is idle, which interrupted long runs several times. Checkpointing and a restart script now make that cost at most one step, but multi-hour runs belong on a dedicated machine or GPU. The GPU path now runs: on an RTX 4070 SUPER the full test suite passes with CuPy, and a fused CUDA kernel runs the 3-D phase field 3.6× (float64) to 46× (float32) faster than 20 CPU cores."));

// Appendix
body.push(H1("Appendix A. Reproducing the results"));
const cmds = [
  "pytest -q tests/test_freesurface.py tests/test_powder.py tests/test_defects.py tests/test_phasefield.py tests/test_phasefield3d.py tests/test_eds.py tests/test_fcc_faults.py tests/test_hcp_kinetics.py tests/test_grain_ca.py",
  "python tools/lbm/freesurface_caseB.py marangoni 10",
  "python benchmarks/literature/porosity_maps.py run IN625 30 ; python benchmarks/literature/porosity_maps.py figure",
  "python benchmarks/literature/pf_path_trapping.py run <tag> ... ; ... report ; ... figure",
  "python benchmarks/literature/pf3d_in625_eds.py run 2d_W10 ; ... run 3d_W10 ; ... summarise 3d_W10 ; ... figure",
  "python benchmarks/literature/nist_ambench2018_tracks_ca.py section ; ... run ; ... figure",
  "python benchmarks/literature/ca_multilayer.py run 90 480 1 ; ... run 67 480 1 ; ... figure",
  "python benchmarks/literature/cantor_gyroid_ca.py section ; ... run thin ; ... run thick ; ... run thin_a1 ; ... figure",
  "python benchmarks/literature/cantor_gyroid_faults.py run ; FAULTS_VOX=0.05 python benchmarks/literature/cantor_gyroid_faults.py run",
  "python benchmarks/literature/femncocr_gyroid_hcp.py run",
  "python benchmarks/literature/cantor_gyroid_faults_asbuilt.py build ; ... compress ; ... figure",
];
cmds.forEach((c) => body.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: c, font: "Consolas", size: 17 })] })));
body.push(P("Results are stored as JSON and NPZ files in benchmarks/literature/results/; figures in docs/img/."));

// ---------- document ----------
const doc = new Document({
  creator: "TPMSsuite",
  title: "TPMSsuite process and microstructure results",
  description: "Melt-pool flow, porosity, solute trapping, 3-D multicomponent cells with synthetic EDS maps, 3-D grain structure",
  styles: {
    default: { document: { run: { font: FONT, size: 20, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 21, bold: true, color: INK },
        paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 260 } } } } ] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 300 } } } } ] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "TPMSsuite: process and microstructure results, November 2026", font: FONT, size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
