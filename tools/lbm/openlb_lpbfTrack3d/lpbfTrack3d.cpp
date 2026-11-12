/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * This file is a standalone OpenLB application. Its collision operator is adapted from
 * OpenLB's TotalEnthalpyAdvectionDiffusionBGKdynamics (GPL-2.0-or-later), so this file is
 * distributed under the GPL-2.0-or-later, unlike the rest of tpmsfem (Apache-2.0). It is not
 * linked into tpmsfem: it runs as a separate program and writes a binary field file.
 */
/* AM-Bench 2018-02 case-B laser track as conduction + melting in OpenLB (feasibility check
 * for tpmsfem). Total-enthalpy LBM (Huang & Wu 2015, OpenLB TotalEnthalpyAdvectionDiffusionBGK)
 * on D3Q7, extended with a volumetric laser source: after collision each population gets
 * t_i * S, S = q dt / (rho cp) (enthalpy in kelvin with cp = 1).
 * Units: temperature in C, enthalpy in K (H = T + L/cp * liquid fraction), lengths in m.
 * Output: the final enthalpy field at the cell centres of the tpmsfem grid (binary doubles). */
#include <olb.h>

using namespace olb;
using namespace olb::names;

using DESC   = descriptors::D3Q7<descriptors::VELOCITY, descriptors::SOURCE>;
using MyCase = Case<Temperature, Lattice<double, DESC>>;
using T      = double;

// material (constant properties, IN625-like) and laser (case B, calibrated A and depth)
constexpr T RHO = 8440., CP = 650., K = 25., LAT = 2.72e5, TS = 1290., TL = 1350., T0 = 25.;
constexpr T P = 179.2, ABSORB = 0.3234, W = 85e-6, DELTA = 15.3e-6, V = 0.8;
constexpr T LX = 1.0e-3, LY = 0.3e-3, LZ = 0.15e-3, X0 = 0.15e-3, X1 = 0.85e-3;
constexpr T TAU = 1.0;
// note: the Gaussian is normalised analytically (2/(pi W^2)); on a 5 um grid with W = 85 um the
// discrete sum differs by < 1e-6

template <typename T_, typename DESCRIPTOR, typename MOMENTA = momenta::AdvectionDiffusionBulkTuple>
struct LaserTotalEnthalpyBGK final : public dynamics::CustomCollision<T_, DESCRIPTOR, MOMENTA> {
  static constexpr bool is_vectorizable = false;
  using MomentaF     = typename MOMENTA::template type<DESCRIPTOR>;
  using EquilibriumF = typename equilibria::FirstOrder::template type<DESCRIPTOR, MOMENTA>;
  using parameters   = meta::list<descriptors::OMEGA, TotalEnthalpy::T_S, TotalEnthalpy::T_L, TotalEnthalpy::CP_S,
                                  TotalEnthalpy::CP_L, TotalEnthalpy::LAMBDA_S, TotalEnthalpy::LAMBDA_L, TotalEnthalpy::L>;
  template <typename NEW_T> using exchange_value_type = LaserTotalEnthalpyBGK<NEW_T, DESCRIPTOR, MOMENTA>;
  template <typename NEW_D> using exchange_descriptor = LaserTotalEnthalpyBGK<T_, NEW_D, MOMENTA>;
  template <typename M> using exchange_momenta = LaserTotalEnthalpyBGK<T_, DESCRIPTOR, M>;
  std::type_index id() override { return typeid(LaserTotalEnthalpyBGK); }
  AbstractParameters<T_, DESCRIPTOR>& getParameters(BlockLattice<T_, DESCRIPTOR>& block) override {
    return block.template getData<OperatorParameters<LaserTotalEnthalpyBGK>>();
  }
  void computeEquilibrium(ConstCell<T_, DESCRIPTOR>& cell, T_ rho, const T_ u[DESCRIPTOR::d], T_ fEq[DESCRIPTOR::q]) const override {
    for (int i = 0; i < DESCRIPTOR::q; i++) fEq[i] = equilibrium<DESCRIPTOR>::firstOrder(i, rho, u);
  }
  template <typename CELL, typename PARAMETERS, typename V = typename CELL::value_t>
  CellStatistic<V> collide(CELL& cell, PARAMETERS& parameters) any_platform {
    using namespace TotalEnthalpy;
    const V cp_s = parameters.template get<CP_S>(), cp_l = parameters.template get<CP_L>();
    const V T_s = parameters.template get<T_S>(), T_l = parameters.template get<T_L>(), l = parameters.template get<L>();
    const V lambda_s = parameters.template get<LAMBDA_S>(), lambda_l = parameters.template get<LAMBDA_L>();
    const V cp_ref = V{2} * cp_s * cp_l / (cp_s + cp_l);
    const V H = MomentaF().computeRho(cell);
    const V H_s = cp_s * T_s, H_l = cp_l * T_l + l;
    V temperature, fl;
    if (H <= H_s) { temperature = T_s - (H_s - H) / cp_s; fl = 0; }
    else if (H >= H_l) { temperature = T_l + (H - H_l) / cp_l; fl = 1; }
    else { temperature = (H_l - H) / (H_l - H_s) * T_s + (H - H_s) / (H_l - H_s) * T_l; fl = (H - H_s) / l; }
    const V lambda = (V{1} - fl) * lambda_s + fl * lambda_l;
    const V cp = (V{1} - fl) * cp_s + fl * cp_l;
    const V omega = V{1} / (lambda / cp_ref * descriptors::invCs2<T_, DESCRIPTOR>() + V{0.5});
    const V S = cell.template getField<descriptors::SOURCE>();
    // zero velocity: conduction only
    for (int i = 0; i < DESCRIPTOR::q; ++i) {
      V feq = cp * temperature * descriptors::t<T_, DESCRIPTOR>(i) * (cp_ref / cp) - descriptors::t<T_, DESCRIPTOR>(i);
      if (i == 0) feq += H - cp_ref * temperature;
      cell[i] = cell[i] * (V{1} - omega) + omega * feq + descriptors::t<T_, DESCRIPTOR>(i) * S;
    }
    return {H, V{0}};
  }
  std::string getName() const override { return "LaserTotalEnthalpyBGK"; }
};

int main(int argc, char* argv[]) {
  initialize(&argc, &argv);
  OstreamManager clout(std::cout, "main");
  const int  res = argc > 1 ? std::atoi(argv[1]) : 5;          // cell size in um
  const T    dx  = res * 1e-6;
  const T    alpha = K / (RHO * CP);
  const T    cs2 = T{1} / descriptors::invCs2<T, DESC>();
  const T    dt  = cs2 * (TAU - 0.5) * dx * dx / alpha;

  MyCase::ParametersD params;
  params.set<parameters::OVERLAP>(1);
  // nodes at the cell centres of the tpmsfem grid; bounce-back walls half a cell outside them
  Vector<T, 3> extent(LX + dx, LY + dx, LZ + dx), origin(-dx / 2, -dx / 2, -dx / 2);
  IndicatorCuboid3D<T> box(extent, origin);
  Mesh<T, 3> mesh(box, dx, 1);
  mesh.setOverlap(1);
  MyCase myCase(params, mesh);
  auto& geometry = myCase.getGeometry();
  geometry.rename(0, 2);
  geometry.rename(2, 1, {1, 1, 1});
  geometry.clean();
  geometry.checkForErrors();
  auto& lat = myCase.getLattice(Temperature{});
  lat.setUnitConverter<AdeUnitConverter<T, DESC>>(dx, dt, LX, T{1}, alpha, RHO);

  dynamics::set<LaserTotalEnthalpyBGK>(lat, geometry.getMaterialIndicator({1}));
  boundary::set<boundary::BounceBack>(lat, geometry, 2);          // adiabatic walls
  const T lam = T{1} * cs2 * (TAU - 0.5);
  lat.setParameter<descriptors::OMEGA>(T{1} / TAU);
  lat.setParameter<TotalEnthalpy::T_S>(TS);
  lat.setParameter<TotalEnthalpy::T_L>(TL);
  lat.setParameter<TotalEnthalpy::CP_S>(T{1});
  lat.setParameter<TotalEnthalpy::CP_L>(T{1});
  lat.setParameter<TotalEnthalpy::LAMBDA_S>(lam);
  lat.setParameter<TotalEnthalpy::LAMBDA_L>(lam);
  lat.setParameter<TotalEnthalpy::L>(LAT / CP);
  std::vector<T> zero(3, T{});
  AnalyticalConst3D<T, T> u(zero);
  AnalyticalConst3D<T, T> H0(T0);
  AnalyticalConst3D<T, T> S0(T{0});
  fields::set<descriptors::VELOCITY>(lat, geometry.getMaterialIndicator({1, 2}), u);
  fields::set<descriptors::SOURCE>(lat, geometry.getMaterialIndicator({1, 2}), S0);
  lat.iniEquilibrium(geometry.getMaterialIndicator({1, 2}), H0, u);
  lat.initialize();

  // laser: q(x,y,z) = A P 2/(pi W^2) exp(-2 r^2/W^2) * exp(-d/delta)/(delta (1-e^-6)), d = LZ - z < 6 delta
  // normalised on the grid like tpmsfem: depth weights at the node depths d = (k + 1/2) dx < 6 delta
  T depthSum = 0;
  for (int k = 0; (k + 0.5) * dx <= 6 * DELTA; ++k) depthSum += std::exp(-(k + 0.5) * dx / DELTA) * dx;
  const T q0 = ABSORB * P * 2. / (M_PI * W * W) / depthSum;
  const T Sfac = dt / (RHO * CP);
  const T tEnd = (X1 - X0) / V;
  const std::size_t nSteps = std::size_t(std::ceil(tEnd / dt));
  clout << "dx = " << dx << " m, dt = " << dt << " s, steps = " << nSteps << std::endl;
  util::Timer<T> timer(nSteps, geometry.getStatistics().getNvoxel());
  timer.start();
  for (std::size_t iT = 0; iT < nSteps; ++iT) {
    const T xl = X0 + V * (iT + 0.5) * dt, yl = LY / 2;
    AnalyticalFfromCallableF<3, T, T> src([&](Vector<T, 3> x) -> Vector<T, 1> {
      const T d = LZ - x[2];
      if (d < 0 || d > 6 * DELTA + 1e-12) return T{0};
      const T r2 = (x[0] - xl) * (x[0] - xl) + (x[1] - yl) * (x[1] - yl);
      if (r2 > 9 * W * W) return T{0};
      return q0 * std::exp(-2 * r2 / (W * W)) * std::exp(-d / DELTA) * Sfac;
    });
    fields::set<descriptors::SOURCE>(lat, geometry.getMaterialIndicator({1}), src);
    lat.collideAndStream();
    if (iT % 200 == 0) timer.print(iT);
  }
  timer.stop();
  timer.printSummary();

  // final enthalpy at cell centres of the tpmsfem grid (x fastest)
  lat.setProcessingContext(ProcessingContext::Evaluation);
  SuperLatticeDensity3D<T, DESC> Hf(lat);
  AnalyticalFfromSuperF3D<T> Hi(Hf, true);
  const int nx = int(std::round(LX / dx)), ny = int(std::round(LY / dx)), nz = int(std::round(LZ / dx));
  std::vector<double> out(std::size_t(nx) * ny * nz);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        T x[3] = {(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx}, h[1] = {0};
        Hi(h, x);
        out[(std::size_t(k) * ny + j) * nx + i] = h[0];
      }
  std::ofstream f("enthalpy.bin", std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(double));
  clout << "wrote enthalpy.bin " << nx << " x " << ny << " x " << nz << std::endl;
  return 0;
}
