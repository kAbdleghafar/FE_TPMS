// Word document: critical self-review of the 3-D multicomponent phase field and synthetic EDS results.
// Usage: node tools/report/make_review_3d_eds.js docs/reports/<name>.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, Footer, Header, PageNumber,
} = require("docx");

const OUT = process.argv[2];
const FONT = "Arial";
const TEXT_W = 9638; // A4, 2 cm margins (DXA)
const INK = "1F1F1F", MUTED = "5A5A5A", ACCENT = "1F4E79", HEAD_FILL = "DCE6F0", ROW_FILL = "F4F6F8";
const STRONG = "B42318", MODERATE = "B54708", FAIR = "1F4E79";

function runs(text, base = {}) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter((p) => p.length);
  return parts.map((p) => p.startsWith("**") ? new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, ...base })
                                             : new TextRun({ text: p, font: FONT, ...base }));
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text),
                                               spacing: { after: 60, line: 264 } });

let tabNo = 0;
const tcap = (text) => { tabNo += 1; return new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: [
  new TextRun({ text: `Table ${tabNo}. `, bold: true, font: FONT, size: 18, color: MUTED }), ...runs(text, { size: 18, color: MUTED })] }); };

function table(headers, rows, widths, colour = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round(x * TEXT_W / total));
  w[w.length - 1] += TEXT_W - w.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, header, shade) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, borders,
    shading: header ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" }
                    : (shade ? { fill: ROW_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: runs(String(text), { size: 16, bold: header,
      color: (!header && colour[i] && colour[i][text]) ? colour[i][text] : INK }) })],
  });
  const trs = [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true, false)) })];
  rows.forEach((r, k) => trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, k % 2 === 1)) })));
  return [new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: w, rows: trs }),
          new Paragraph({ children: [], spacing: { after: 120 } })];
}
const strengthColour = { 2: { "strong": STRONG, "moderate": MODERATE, "fair": FAIR } };

// ------------------------------------------------------------------ content
const body = [];
body.push(
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "TPMSsuite", font: FONT, size: 24, color: MUTED })] }),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({
    text: "3-D multicomponent phase field and synthetic EDS maps: points a reviewer could raise", font: FONT, size: 36, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 60 }, children: runs("Critical self-review of Section 6 of the progress report (revision 2, 16 November 2026) and of docs/process_models.md §5.6.", { size: 20, color: MUTED }) }),
  new Paragraph({ spacing: { after: 60 }, children: runs("16 November 2026 · Khaled Abdelghafar and Claude (Anthropic) · repository kAbdleghafar/TPMSsuite", { size: 20, color: MUTED }) }),
  new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "BFBFBF", space: 4 } }, children: [] }),
);

body.push(H1("1. Verdict"));
body.push(P("A reviewer cannot reasonably argue that the code solves the wrong equations: the verification evidence below is strong. A reviewer can, with justification, argue that **several of the reported numbers are not yet final**: the analysed material is partly unsolidified, the cell spacing depends on the run history, the 3-D resolution is not converged, the SEM numbers rest on an assumed shape of the X-ray generation volume, and the experimental anchor is weak."));
body.push(P("**Recommendation.** Treat the qualitative findings as defensible and the exact numbers (0.33 µm spacing, 27 % SEM contrast, the droplet compositions) as preliminary until the checks in Section 6 are done; soften the wording as in Section 5 in anything shared before then."));

body.push(H1("2. What is not in question"));
body.push(
  B("**The 3-D code reduces to the verified 2-D model.** With one solute and a box one cell thick it reproduces the 2-D phase field to 1e-15 over 300 steps (tests/test_phasefield3d.py)."),
  B("**The multicomponent coupling is consistent.** Splitting a solute into two identical halves changes nothing (to 1e-13)."),
  B("**The two kernels agree** (numba CPU and NumPy/CuPy arrays, to 1e-12, rotated crystal, four solutes), and **every solute is conserved to round-off** in a closed box."),
  B("**The 2-D model was verified** against Mullins–Sekerka linear stability and against the published phase field of Ghosh et al. (2017, 2018), and its spacing converged within 8 % in the interface width at this point (W0 = 10, 6.5, 4.1 nm)."),
  B("**The EDS implementation does what is stated:** the binned depth integration matches the continuous transfer function within 5 %, a point source keeps 90 % of its signal within the generation range, and the counting statistics have the expected mean (tests/test_eds.py)."),
);

body.push(H1("3. Legitimate criticisms"));
body.push(tcap("Summary. Strength: how hard the point would be to rebut today."));
body.push(...table(["#", "criticism", "strength", "what we have now", "what would settle it"], [
  ["C1", "The analysed sections are in the mushy zone, not the final solid", "strong", "a statement in the code documentation only", "taller window or extended run; report the liquid fraction of every section; drop the droplet maxima"],
  ["C2", "The cell spacing depends on the run history (still coarsening; artificial flat start)", "strong", "reported as a lower bound", "grow along the real G–R history from the fusion boundary, or run until the spacing saturates"],
  ["C3", "The interface width is not converged in 3-D (D/V = 3.3 W0)", "moderate", "2-D convergence series at the same point", "3-D run at W0 = 6.5 nm on a smaller box"],
  ["C4", "One random seed, one periodic box (about 24 cells)", "moderate", "none", "second seed; 2.4 µm box; scatter band"],
  ["C5", "The SEM contrast rests on an assumed generation volume", "strong", "sensitivity stated (27 % vs < 1 % for one Gaussian)", "Monte Carlo electron trajectories; secondary fluorescence; a measured line scan"],
  ["C6", "The Cr and Fe maps restate the input partition coefficients; 'below the noise' depends on the counts chosen (and the Fe variation was misstated)", "fair", "inputs sourced; Fe statement corrected", "present as input-driven; give the counts needed to detect them"],
  ["C7", "The experimental anchor is weak (a five-fold spacing range)", "strong", "measured range 0.2–1.0 µm only", "measured STEM-EDS or atom-probe profiles across cell walls of LPBF IN625; local spacing statistics of case B"],
  ["C8", "Model simplifications (dilute linear alloy, estimated Mo slope, one diffusivity, assumed V_D, aligned crystal)", "moderate", "stated in the docs", "sensitivity runs on the uncertain inputs"],
], [0.5, 3.0, 1.0, 2.4, 3.1], strengthColour));

body.push(H2("C1. The analysed sections are in the mushy zone"));
body.push(
  B("**What a reviewer would say.** The 'solidified' microstructure is a snapshot of the mushy zone just behind the cell tips, not the final solid."),
  B("**Why it is valid.** Rows are stored when they leave the bottom quarter of the 1.2 µm window, 0.4–0.7 µm behind the tips. At the end of the run that part of the window was **7–10 % liquid** (phi < 0). The transverse section analysed for EDS (0.8 µm below the tips) consists of rows stored this way. The stored liquid keeps its composition; its later solidification, with further enrichment and the Laves / NbC eutectic of IN625, is not followed. The dilute linear model has no eutectic, so its last liquid never solidifies completely."),
  B("**Consequences.** The wall-centre compositions and the 95th percentiles are probably underestimated; the droplet values (13–21 wt% Nb) are not final compositions and should not be quoted. SEM maps are little affected (they average over a volume much larger than the redistribution), STEM wall profiles would change."),
  B("**Fix.** A taller window (so stored rows are fully solid, roughly proportional extra cost), sections taken only where the solid fraction exceeds 99 %, and the liquid fraction reported for every section; an approximate Scheil closure of the remaining liquid as an option."),
);
body.push(H2("C2. The spacing depends on the run history"));
body.push(
  B("**Evidence of the transient.** Spacing (sqrt(A/N)) against the time the material was grown before the end of the run: 0.24 µm (5 µm earlier), 0.27 (4 µm), 0.28 (3 µm), 0.30 (2 µm), 0.31 (1.5 µm), 0.32 (1 µm), 0.33 µm (0.6 µm). The tips were still moving ahead of the T0 isotherm: 1.23 µm at 3.1 µm pulled, 1.44 µm at 5 µm."),
  B("**The start is artificial.** A flat front with 2 % noise at fixed G and R. In the real pool the cells grow up from the fusion boundary while R rises from 0.027 to 0.092 m/s over the first 10 µm."),
  B("**Fix.** Make G and R time-dependent along the case-B path (a small change to the frozen-temperature term) and grow from the fusion boundary; or run until the spacing saturates. Report a band of spacings rather than one number."),
);
body.push(H2("C3. Interface width"));
body.push(
  B("W0 = 10 nm gives D/V = 32.6 nm = 3.3 W0, below the 5 W0 used for the 2-D path study and well below the 10 W0 usually recommended."),
  B("The 2-D series at this point (W0 = 10 / 6.5 / 4.1 nm) gave spacings 0.35 / 0.32 / 0.33 µm, cores 3.47–3.50 wt% Nb and tip liquid 5.24–5.72 wt%: about 6–9 % on spacing and tip liquid. This has not been checked in 3-D."),
  B("**Fix.** A 0.8 × 0.8 µm box at W0 = 6.5 nm against 10 nm (about 6 CPU-hours here, minutes on a GPU)."),
);
body.push(H2("C4. One realisation, one box"));
body.push(
  B("A single seed and a periodic 1.6 × 1.6 µm box holding 22–25 cells near the tips. One cell more or less changes sqrt(A/N) by about 2 %, and periodic boundaries can stabilise an array that would otherwise coarsen."),
  B("**Fix.** A second seed (2.2 h) and a 2.4 µm box (about 5 h); quote mean and range."),
);
body.push(H2("C5. The SEM generation volume is assumed"));
body.push(
  B("The volume is depth-resolved (lateral width growing as depth^1.5) and calibrated so that 90 % of the signal lies within the Anderson–Hasler range. With it, SEM at 15 kV keeps 27 % of the Nb contrast; a single Gaussian with the same 90 % diameter keeps under 1 %. The exact figure is therefore model-dependent."),
  B("Not included: matrix (ZAF) corrections, and **secondary fluorescence**, by which Ni Kα excites Cr K and Fe K from a much larger volume; it lowers the Cr and Fe contrast further."),
  B("**Fix.** Electron-trajectory Monte Carlo for the generation volume (an own single-scattering implementation, or NIST DTSA-II), with fluorescence; calibration against a measured line scan across a known feature."),
);
body.push(H2("C6. Cr and Fe are input-driven"));
body.push(
  B("Their pattern follows from k = 1.02 and 1.09 (Gan et al. 2019) and from the choice that they do not affect the interface temperature. In the analysed section the 5–95 % spread is **2.2 % of the content for Cr and 9 % for Fe**."),
  B("**Correction.** The earlier statement that 'Cr and Fe vary by under 2 % of their content' (docs §5.6, report Section 6.5) holds for Cr but not for Fe; it has been corrected. Fe is still undetectable in these maps, but because of its low content (0.7 wt%), not a small relative variation."),
  B("Whether it is 'below the noise' depends on the acquisition. With 1500 counts per pixel for pure element (SEM, 15 kV): Cr gives about 310 counts (5–6 % noise) against a 2 % signal; Fe gives about 11 counts (about 30 % noise) against a 9 % signal. Detecting the Cr difference at 3 sigma needs about (3 / 0.022)² ≈ 19,000 Cr counts per point, the Fe difference about (3 / 0.09)² ≈ 1,100 Fe counts (about 100 pixels binned): both reachable in long line scans or point analyses, not in a typical map."),
  B("**Fix.** State the result as a consequence of the input k values and the map statistics, with the counts needed for detection."),
);
body.push(H2("C7. Weak experimental anchor"));
body.push(
  B("The only experimental comparison is the case-B spacing range measured by SEM, 0.2–1.0 µm (Ghosh et al. 2018). Agreement with a five-fold range is not validation; no measured composition profile has been compared."),
  B("**Fix.** Measured STEM-EDS or atom-probe profiles across cell walls in LPBF IN625 from the literature (wall and core Nb and Mo); local spacing statistics from NIST's case-B micrographs; ideally new STEM-EDS on AM-Bench material."),
);
body.push(H2("C8. Model simplifications"));
body.push(
  B("Dilute alloy with linear liquidus surfaces; Mo liquidus slope estimated from the Ni–Mo binary (−2.9 K/wt%, Mo carries 10 % of the freezing range); one liquid diffusivity for all solutes; V_D = 1 m/s assumed for trapping; frozen temperature field without latent heat; no solid diffusion; one crystal orientation, <100> along the growth direction; eps4 = 0.02."),
  B("These are standard in the field, but a reviewer can ask how much each matters. **Fix.** One-at-a-time sensitivity runs (Mo slope ±50 %, V_D 0.5–2 m/s, a tilted crystal); CALPHAD liquidus slopes if a database is available."),
);

body.push(H1("4. What should hold up regardless"));
body.push(
  B("**The structure:** columnar cells with Nb and Mo enriched at the walls and most strongly at the triple junctions; Cr, Fe and Ni slightly depleted there. This follows from k < 1 for Nb and Mo at these growth conditions and agrees qualitatively with published STEM-EDS of LPBF IN625."),
  B("**The spacing scale,** about 0.3 µm: inside the measured range and supported by the 2-D convergence series; only the exact value is open."),
  B("**SEM-EDS at 15 kV cannot quantify cell-wall segregation:** its generation volume (about 0.7 µm) is twice the cell spacing, so any reasonable volume shape loses most of the contrast. The direction is robust; the magnitude is not."),
  B("**2-D and 3-D give similar segregation statistics** (within 0.2 wt%): a model-internal comparison, useful for choosing the cheaper 2-D model, not an independent validation."),
);

body.push(H1("5. Wording to soften now"));
body.push(tcap("Claims as written in the report and docs, and a defensible form until the checks are done."));
body.push(...table(["current wording", "defensible wording"], [
  ["The 3-D spacing is 0.33 µm", "about 0.3 µm; 0.33 µm in the material grown last, still coarsening (a lower bound for this history)"],
  ["SEM-EDS at 15 kV keeps 27 % of the Nb contrast", "SEM-EDS at 15 kV keeps a small fraction of the Nb contrast (27 % with our generation-volume model; < 1 % with a single Gaussian)"],
  ["Last-liquid droplets reach 13–21 wt% Nb", "remove, or: intercellular liquid behind the tips is strongly enriched; its final composition is not modelled"],
  ["The segregation matches the 2-D model within 0.2 wt%", "in the mushy zone 0.4–1 µm behind the tips, 2-D and 3-D statistics agree within 0.2 wt%"],
  ["Cr and Fe vary by under 2 % of their content (corrected)", "Cr varies by about 2 % and Fe by about 9 % of its 0.7 wt% (from their k values); both are below the noise of a typical map, Fe because of its low content; long line scans could detect them"],
  ["The cells lie inside the measured range", "the spacing lies inside the 0.2–1.0 µm range measured by SEM; composition profiles have not been compared with experiment"],
], [4, 6]));

body.push(H1("6. Checks before submission"));
body.push(tcap("Planned checks, in order; costs on the 4-core cloud machine unless stated."));
body.push(...table(["#", "check", "addresses", "cost"], [
  ["1", "Taller window or extended run; sections only where solid fraction > 99 %; liquid fraction reported; EDS re-run", "C1", "a few CPU-hours"],
  ["2", "Monte Carlo electron trajectories (with secondary fluorescence) for the SEM generation volume", "C5, C6", "about a day of coding; minutes per case"],
  ["3", "3-D interface-width check: 0.8 µm box at W0 = 10 and 6.5 nm", "C3", "about 6 CPU-hours"],
  ["4", "Second seed; 2.4 µm box", "C4", "2–5 CPU-hours each"],
  ["5", "Time-dependent G and R along the case-B path from the fusion boundary", "C2", "small code change; about 2× the present run"],
  ["6", "Literature search for measured cell-wall compositions and spacings in LPBF IN625 (STEM-EDS, atom probe)", "C7", "a few hours"],
  ["7", "Sensitivity runs: Mo slope, V_D, crystal tilt", "C8", "2–3 CPU-hours each (2-D runs suffice)"],
], [0.5, 5.6, 1.2, 2.7]));
body.push(P("Checks 1, 3 and 4 run unattended on the present machine; a GPU would bring the converged 3-D runs down from hours to minutes."));

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "TPMSsuite",
  title: "3-D phase field and EDS: points a reviewer could raise",
  description: "Critical self-review of the 3-D multicomponent phase-field and synthetic EDS results",
  styles: {
    default: { document: { run: { font: FONT, size: 20, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 320, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 260 } } } } ] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "TPMSsuite: 3-D phase field and EDS, critical self-review", font: FONT, size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
