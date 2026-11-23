/* Standalone C99 DOP853 integrator and cycle-Fourier kernel for Lorenz-63.
 *
 * Implements SciPy's DOP853 tableau, Hairer step-size control, and lazy 7th-order
 * dense output directly on the 3D forced Lorenz system, with fused cycle-grid
 * sampling and discrete Fourier projection to avoid trajectory allocations.
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DOP853_SAFETY 0.9
#define DOP853_MIN_FACTOR 0.2
#define DOP853_MAX_FACTOR 10.0
#define DOP853_ERROR_EXPONENT (-0.125)

static const double DOP_C[16] = {
    0.0,
    0.05260015195876773,
    0.0789002279381516,
    0.1183503419072274,
    0.2816496580927726,
    0.3333333333333333,
    0.25,
    0.3076923076923077,
    0.6512820512820513,
    0.6,
    0.8571428571428571,
    1.0,
    1.0,
    0.1,
    0.2,
    0.7777777777777778
};

static const double DOP_B[12] = {
    0.054293734116568765,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145003,
    -5.801203960010585,
    0.3111643669578199,
    -0.1521609496625161,
    0.20136540080403034,
    0.04471061572777259
};

static const double DOP_E3[13] = {
    -0.18980075407240762,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145003,
    -5.801203960010585,
    -0.4226823213237919,
    -0.1521609496625161,
    0.20136540080403034,
    0.02265179219836082,
    0.0
};

static const double DOP_E5[13] = {
    0.01312004499419488,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.2251564463762044,
    -0.4957589496572502,
    1.6643771824549864,
    -0.35032884874997366,
    0.3341791187130175,
    0.08192320648511571,
    -0.022355307863886294,
    0.0
};

static const double DOP_A[16][16] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.05260015195876773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0197250569845379, 0.0591751709536137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.02958758547680685, 0.0, 0.08876275643042054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.2413651341592667, 0.0, -0.8845494793282861, 0.924834003261792, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.037037037037037035, 0.0, 0.0, 0.17082860872947386, 0.12546768756682242, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.037109375, 0.0, 0.0, 0.17025221101954405, 0.06021653898045596, -0.017578125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.03709200011850479, 0.0, 0.0, 0.17038392571223998, 0.10726203044637328, -0.015319437748624402, 0.008273789163814023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.6241109587160757, 0.0, 0.0, -3.3608926294469414, -0.868219346841726, 27.59209969944671, 20.154067550477894, -43.48988418106996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.47766253643826434, 0.0, 0.0, -2.4881146199716677, -0.590290826836843, 21.230051448181193, 15.279233632882423, -33.28821096898486, -0.020331201708508627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {-0.9371424300859873, 0.0, 0.0, 5.186372428844064, 1.0914373489967295, -8.149787010746927, -18.52006565999696, 22.739487099350505, 2.4936055526796523, -3.0467644718982196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {2.273310147516538, 0.0, 0.0, -10.53449546673725, -2.0008720582248625, -17.9589318631188, 27.94888452941996, -2.8589982771350235, -8.87285693353063, 12.360567175794303, 0.6433927460157636, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, 0.3111643669578199, -0.1521609496625161, 0.20136540080403034, 0.04471061572777259, 0.0, 0.0, 0.0, 0.0},
    {0.056167502283047954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25350021021662483, -0.2462390374708025, -0.12419142326381637, 0.15329179827876568, 0.00820105229563469, 0.007567897660545699, -0.008298, 0.0, 0.0, 0.0},
    {0.03183464816350214, 0.0, 0.0, 0.0, 0.0, 0.028300909672366776, 0.053541988307438566, -0.05492374857139099, 0.0, 0.0, -0.00010834732869724932, 0.0003825710908356584, -0.00034046500868740456, 0.1413124436746325, 0.0, 0.0},
    {-0.42889630158379194, 0.0, 0.0, 0.0, 0.0, -4.697621415361164, 7.683421196062599, 4.06898981839711, 0.3567271874552811, 0.0, 0.0, 0.0, -0.0013990241651590145, 2.9475147891527724, -9.15095847217987, 0.0}
};

static const double DOP_D[4][16] = {
    {-8.428938276109013, 0.0, 0.0, 0.0, 0.0, 0.5667149535193777, -3.0689499459498917, 2.38466765651207, 2.117034582445028, -0.871391583777973, 2.2404374302607883, 0.6315787787694688, -0.08899033645133331, 18.148505520854727, -9.194632392478356, -4.436036387594894},
    {10.427508642579134, 0.0, 0.0, 0.0, 0.0, 242.28349177525817, 165.20045171727028, -374.5467547226902, -22.113666853125306, 7.733432668472264, -30.674084731089398, -9.332130526430229, 15.697238121770845, -31.139403219565178, -9.35292435884448, 35.81684148639408},
    {19.985053242002433, 0.0, 0.0, 0.0, 0.0, -387.0373087493518, -189.17813819516758, 527.8081592054236, -11.57390253995963, 6.8812326946963, -1.0006050966910838, 0.7777137798053443, -2.778205752353508, -60.19669523126412, 84.32040550667716, 11.99229113618279},
    {-25.69393346270375, 0.0, 0.0, 0.0, 0.0, -154.18974869023643, -231.5293791760455, 357.6391179106141, 93.40532418362432, -37.45832313645163, 104.0996495089623, 29.8402934266605, -43.53345659001114, 96.32455395918828, -39.17726167561544, -149.72683625798564}
};

typedef struct {
    double sigma;
    double rho;
    double beta;
    double omega;
    double forcing[3];
    double phase[3];
    int has_forcing;
} LorenzParams;

static inline void lorenz_rhs(double t, const double y[3], const LorenzParams *p, double out[3]) {
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (p->has_forcing) {
        if (p->forcing[0] != 0.0) {
            fx = p->forcing[0] * sin(p->omega * t + p->phase[0]);
        }
        if (p->forcing[1] != 0.0) {
            fy = p->forcing[1] * sin(p->omega * t + p->phase[1]);
        }
        if (p->forcing[2] != 0.0) {
            fz = p->forcing[2] * sin(p->omega * t + p->phase[2]);
        }
    }
    out[0] = p->sigma * (y[1] - y[0]) + fx;
    out[1] = p->rho * y[0] - y[1] - y[0] * y[2] + fy;
    out[2] = y[0] * y[1] - p->beta * y[2] + fz;
}

static double select_initial_step(
    const LorenzParams *p,
    double t0,
    const double y0[3],
    double t_bound,
    double max_step,
    const double f0[3],
    double rtol,
    const double atol[3]
) {
    double interval_length = fabs(t_bound - t0);
    if (interval_length == 0.0) {
        return 0.0;
    }

    double scale[3];
    double d0_sq = 0.0, d1_sq = 0.0;
    for (int j = 0; j < 3; ++j) {
        scale[j] = atol[j] + fabs(y0[j]) * rtol;
        double r0 = y0[j] / scale[j];
        double r1 = f0[j] / scale[j];
        d0_sq += r0 * r0;
        d1_sq += r1 * r1;
    }
    double d0 = sqrt(d0_sq / 3.0);
    double d1 = sqrt(d1_sq / 3.0);

    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : (0.01 * d0 / d1);
    if (h0 > interval_length) {
        h0 = interval_length;
    }

    double y1[3], f1[3];
    for (int j = 0; j < 3; ++j) {
        y1[j] = y0[j] + h0 * f0[j];
    }
    lorenz_rhs(t0 + h0, y1, p, f1);

    double d2_sq = 0.0;
    for (int j = 0; j < 3; ++j) {
        double r2 = (f1[j] - f0[j]) / scale[j];
        d2_sq += r2 * r2;
    }
    double d2 = sqrt(d2_sq / 3.0) / h0;

    double h1;
    if (d1 <= 1e-15 && d2 <= 1e-15) {
        double cand = h0 * 1e-3;
        h1 = (1e-6 > cand) ? 1e-6 : cand;
    } else {
        double m = (d1 > d2) ? d1 : d2;
        h1 = pow(0.01 / m, 0.125);
    }

    double res = 100.0 * h0;
    if (h1 < res) res = h1;
    if (interval_length < res) res = interval_length;
    if (max_step < res) res = max_step;
    return res;
}

static int dop853_step(
    const LorenzParams *p,
    double *t_ptr,
    double y[3],
    double f[3],
    double *h_abs_ptr,
    double t_bound,
    double max_step,
    double rtol,
    const double atol[3],
    double K[16][3],
    double *h_prev_ptr,
    double y_old[3]
) {
    double t = *t_ptr;
    double min_step = 10.0 * fabs(nextafter(t, INFINITY) - t);
    double h_abs = *h_abs_ptr;
    if (h_abs > max_step) {
        h_abs = max_step;
    } else if (h_abs < min_step) {
        h_abs = min_step;
    }

    int step_accepted = 0;
    int step_rejected = 0;
    double t_new = t;
    double h = h_abs;
    double y_new[3];
    double f_new[3];
    double y_tmp[3];

    while (!step_accepted) {
        if (h_abs < min_step) {
            return -1;
        }

        h = h_abs;
        t_new = t + h;
        if (t_new > t_bound) {
            t_new = t_bound;
        }
        h = t_new - t;
        h_abs = fabs(h);

        for (int j = 0; j < 3; ++j) {
            K[0][j] = f[j];
        }
        for (int s = 1; s < 12; ++s) {
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int m = 0; m < s; ++m) {
                    acc += DOP_A[s][m] * K[m][j];
                }
                y_tmp[j] = y[j] + acc * h;
            }
            lorenz_rhs(t + DOP_C[s] * h, y_tmp, p, K[s]);
        }

        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int m = 0; m < 12; ++m) {
                acc += DOP_B[m] * K[m][j];
            }
            y_new[j] = y[j] + h * acc;
        }
        lorenz_rhs(t + h, y_new, p, f_new);
        for (int j = 0; j < 3; ++j) {
            K[12][j] = f_new[j];
        }

        if (!isfinite(y_new[0]) || !isfinite(y_new[1]) || !isfinite(y_new[2]) ||
            !isfinite(f_new[0]) || !isfinite(f_new[1]) || !isfinite(f_new[2])) {
            return -2;
        }

        double err5_norm_2 = 0.0;
        double err3_norm_2 = 0.0;
        for (int j = 0; j < 3; ++j) {
            double scale = atol[j] + fmax(fabs(y[j]), fabs(y_new[j])) * rtol;
            double e5 = 0.0;
            double e3 = 0.0;
            for (int m = 0; m < 13; ++m) {
                e5 += K[m][j] * DOP_E5[m];
                e3 += K[m][j] * DOP_E3[m];
            }
            e5 /= scale;
            e3 /= scale;
            err5_norm_2 += e5 * e5;
            err3_norm_2 += e3 * e3;
        }

        double error_norm = 0.0;
        if (err5_norm_2 != 0.0 || err3_norm_2 != 0.0) {
            double denom = err5_norm_2 + 0.01 * err3_norm_2;
            error_norm = fabs(h) * err5_norm_2 / sqrt(denom * 3.0);
        }
        if (!isfinite(error_norm)) {
            return -2;
        }

        if (error_norm < 1.0) {
            double factor;
            if (error_norm == 0.0) {
                factor = DOP853_MAX_FACTOR;
            } else {
                factor = DOP853_SAFETY * pow(error_norm, DOP853_ERROR_EXPONENT);
                if (factor > DOP853_MAX_FACTOR) {
                    factor = DOP853_MAX_FACTOR;
                }
            }
            if (step_rejected && factor > 1.0) {
                factor = 1.0;
            }
            h_abs *= factor;
            step_accepted = 1;
        } else {
            double factor = DOP853_SAFETY * pow(error_norm, DOP853_ERROR_EXPONENT);
            if (factor < DOP853_MIN_FACTOR) {
                factor = DOP853_MIN_FACTOR;
            }
            h_abs *= factor;
            step_rejected = 1;
        }
    }

    *h_prev_ptr = h;
    for (int j = 0; j < 3; ++j) {
        y_old[j] = y[j];
        y[j] = y_new[j];
        f[j] = f_new[j];
    }
    *t_ptr = t_new;
    *h_abs_ptr = h_abs;
    return 0;
}

static void dop853_build_dense(
    const LorenzParams *p,
    double t_old,
    double h,
    const double y_old[3],
    const double y_new[3],
    double K[16][3],
    double F[7][3]
) {
    double y_tmp[3];
    for (int s = 13; s < 16; ++s) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int m = 0; m < s; ++m) {
                acc += K[m][j] * DOP_A[s][m];
            }
            y_tmp[j] = y_old[j] + acc * h;
        }
        lorenz_rhs(t_old + DOP_C[s] * h, y_tmp, p, K[s]);
    }

    for (int j = 0; j < 3; ++j) {
        double dy = y_new[j] - y_old[j];
        F[0][j] = dy;
        F[1][j] = h * K[0][j] - dy;
        F[2][j] = 2.0 * dy - h * (K[12][j] + K[0][j]);
        for (int r = 0; r < 4; ++r) {
            double acc = 0.0;
            for (int m = 0; m < 16; ++m) {
                acc += DOP_D[r][m] * K[m][j];
            }
            F[3 + r][j] = h * acc;
        }
    }
}

static inline void dop853_eval_dense(
    double t_q,
    double t_old,
    double h,
    const double y_old[3],
    double F[7][3],
    double val[3]
) {
    double x = (t_q - t_old) / h;
    double one_minus_x = 1.0 - x;
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;
    for (int i = 0; i < 7; ++i) {
        int idx = 6 - i;
        double mult = (i % 2 == 0) ? x : one_minus_x;
        acc0 = (acc0 + F[idx][0]) * mult;
        acc1 = (acc1 + F[idx][1]) * mult;
        acc2 = (acc2 + F[idx][2]) * mult;
    }
    val[0] = acc0 + y_old[0];
    val[1] = acc1 + y_old[1];
    val[2] = acc2 + y_old[2];
}

int lorenz_dop853_spinup_batch(
    const double *y0_batch,
    size_t n_states,
    double spinup_time,
    double sigma,
    double rho,
    double beta,
    double rtol,
    const double atol[3],
    double max_step,
    double first_step,
    double *y_out_batch
) {
    if (!y0_batch || !y_out_batch || !atol || spinup_time < 0.0 ||
        !isfinite(spinup_time) || rtol <= 0.0 || max_step <= 0.0) {
        return -3;
    }
    LorenzParams params = {
        sigma, rho, beta, 0.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0
    };

    for (size_t b = 0; b < n_states; ++b) {
        const double *y0 = y0_batch + b * 3;
        double *y_out = y_out_batch + b * 3;
        double y[3] = {y0[0], y0[1], y0[2]};
        if (!isfinite(y[0]) || !isfinite(y[1]) || !isfinite(y[2])) {
            return -2;
        }
        if (spinup_time == 0.0) {
            y_out[0] = y[0];
            y_out[1] = y[1];
            y_out[2] = y[2];
            continue;
        }

        double f[3];
        lorenz_rhs(0.0, y, &params, f);
        double t = 0.0;
        double h_abs = (first_step > 0.0)
            ? first_step
            : select_initial_step(&params, 0.0, y, spinup_time, max_step, f, rtol, atol);

        double K[16][3];
        double y_old[3];
        double h_prev = 0.0;
        while (t < spinup_time) {
            int rc = dop853_step(
                &params, &t, y, f, &h_abs, spinup_time, max_step,
                rtol, atol, K, &h_prev, y_old
            );
            if (rc != 0) {
                return rc;
            }
        }
        y_out[0] = y[0];
        y_out[1] = y[1];
        y_out[2] = y[2];
    }
    return 0;
}

int lorenz_dop853_phase_samples(
    const double y0[3],
    const double forcing[3],
    const double rhs_phase[3],
    double omega,
    const double *sample_times,
    size_t n_samples,
    double sigma,
    double rho,
    double beta,
    double rtol,
    const double atol[3],
    double max_step,
    double first_step,
    double *values_out
) {
    if (!y0 || !forcing || !rhs_phase || !sample_times || !values_out ||
        !atol || n_samples == 0 || omega <= 0.0 || !isfinite(omega) ||
        rtol <= 0.0 || max_step <= 0.0) {
        return -3;
    }
    for (int j = 0; j < 3; ++j) {
        if (!isfinite(forcing[j]) || !isfinite(rhs_phase[j])) {
            return -3;
        }
    }

    double t_bound = sample_times[n_samples - 1];
    if (!isfinite(t_bound) || t_bound < 0.0) {
        return -3;
    }

    double y[3] = {y0[0], y0[1], y0[2]};
    if (!isfinite(y[0]) || !isfinite(y[1]) || !isfinite(y[2])) {
        return -2;
    }
    if (t_bound == 0.0) {
        for (size_t i = 0; i < n_samples; ++i) {
            values_out[i] = y[0];
            values_out[n_samples + i] = y[1];
            values_out[2 * n_samples + i] = y[2];
        }
        return 0;
    }

    int has_forcing = (forcing[0] != 0.0 || forcing[1] != 0.0 || forcing[2] != 0.0);
    LorenzParams params = {
        sigma, rho, beta, omega,
        {forcing[0], forcing[1], forcing[2]},
        {rhs_phase[0], rhs_phase[1], rhs_phase[2]},
        has_forcing
    };

    double f[3];
    lorenz_rhs(0.0, y, &params, f);
    double t = 0.0;
    double h_abs = (first_step > 0.0)
        ? first_step
        : select_initial_step(&params, 0.0, y, t_bound, max_step, f, rtol, atol);

    double K[16][3];
    double F[7][3];
    double y_old[3];
    double h_prev = 0.0;
    size_t idx = 0;

    while (t < t_bound && idx < n_samples) {
        double t_old = t;
        int rc = dop853_step(
            &params, &t, y, f, &h_abs, t_bound, max_step,
            rtol, atol, K, &h_prev, y_old
        );
        if (rc != 0) {
            return rc;
        }
        if (idx < n_samples && sample_times[idx] <= t) {
            dop853_build_dense(&params, t_old, h_prev, y_old, y, K, F);
            while (idx < n_samples && sample_times[idx] <= t) {
                double val[3];
                dop853_eval_dense(sample_times[idx], t_old, h_prev, y_old, F, val);
                if (!isfinite(val[0]) || !isfinite(val[1]) || !isfinite(val[2])) {
                    return -2;
                }
                values_out[idx] = val[0];
                values_out[n_samples + idx] = val[1];
                values_out[2 * n_samples + idx] = val[2];
                ++idx;
            }
        }
    }
    if (idx != n_samples) {
        return -3;
    }
    return 0;
}

static int lorenz_dop853_fourier_block_impl(
    const double y0[3],
    const double *forcing_vectors,
    const double *forcing_phases,
    int has_forcing_phases,
    size_t n_cond,
    double omega,
    double phase,
    double discard_time,
    size_t n_cycle,
    size_t n_phase,
    const long long *harmonics,
    size_t n_harm,
    double sigma,
    double rho,
    double beta,
    double rtol,
    const double atol[3],
    double max_step,
    double first_step,
    int average_cycles,
    double *out_real,
    double *out_imag,
    double *out_moments
) {
    if (!y0 || !forcing_vectors || !harmonics || !atol || !out_real || !out_imag ||
        n_cond == 0 || n_cycle == 0 || n_phase == 0 || n_harm == 0 ||
        omega <= 0.0 || discard_time < 0.0 || rtol <= 0.0 || max_step <= 0.0) {
        return -3;
    }
    if (has_forcing_phases && !forcing_phases) {
        return -3;
    }
    for (size_t c = 0; c < n_cond * 3; ++c) {
        if (!isfinite(forcing_vectors[c])) {
            return -3;
        }
        if (has_forcing_phases && !isfinite(forcing_phases[c])) {
            return -3;
        }
    }
    if (!isfinite(y0[0]) || !isfinite(y0[1]) || !isfinite(y0[2])) {
        return -2;
    }

    double period = (2.0 * M_PI) / omega;
    size_t total_samples = n_cycle * n_phase;
    double last_time = discard_time + ((double)(n_cycle - 1) + (double)(n_phase - 1) / (double)n_phase) * period;
    if (!isfinite(last_time)) {
        return -3;
    }

    /* Precompute Fourier basis exp(-1j * harmonic * theta[q]) for q = 0..n_phase-1. */
    double *cos_w = (double *)malloc(n_phase * n_harm * sizeof(double));
    double *sin_w = (double *)malloc(n_phase * n_harm * sizeof(double));
    double *cyc_re = (double *)malloc(3 * n_harm * sizeof(double));
    double *cyc_im = (double *)malloc(3 * n_harm * sizeof(double));
    double *mean_re = average_cycles ? (double *)malloc(3 * n_harm * sizeof(double)) : NULL;
    double *mean_im = average_cycles ? (double *)malloc(3 * n_harm * sizeof(double)) : NULL;
    if (!cos_w || !sin_w || !cyc_re || !cyc_im || (average_cycles && (!mean_re || !mean_im))) {
        free(cos_w);
        free(sin_w);
        free(cyc_re);
        free(cyc_im);
        free(mean_re);
        free(mean_im);
        return -3;
    }

    double two_pi = 2.0 * M_PI;
    double first_sample_t = discard_time;
    double raw_angle = omega * first_sample_t + phase;
    double phase_offset = fmod(raw_angle, two_pi);
    if (phase_offset < 0.0) {
        phase_offset += two_pi;
    }
    if (phase_offset >= two_pi) {
        phase_offset -= two_pi;
    }

    for (size_t q = 0; q < n_phase; ++q) {
        double theta = phase_offset + two_pi * (double)q / (double)n_phase;
        for (size_t m = 0; m < n_harm; ++m) {
            double ang = (double)harmonics[m] * theta;
            cos_w[q * n_harm + m] = cos(ang);
            sin_w[q * n_harm + m] = -sin(ang);
        }
    }

    double inv_n_phase = 1.0 / (double)n_phase;
    double inv_n_cycle = 1.0 / (double)n_cycle;
    int status = 0;

    for (size_t cond = 0; cond < n_cond; ++cond) {
        const double *fv = forcing_vectors + cond * 3;
        const double *fp = has_forcing_phases ? (forcing_phases + cond * 3) : NULL;
        LorenzParams params = {
            sigma, rho, beta, omega,
            {fv[0], fv[1], fv[2]},
            {fp ? fp[0] : phase, fp ? fp[1] : phase, fp ? fp[2] : phase},
            (fv[0] != 0.0 || fv[1] != 0.0 || fv[2] != 0.0)
        };

        if (average_cycles) {
            for (size_t k = 0; k < 3 * n_harm; ++k) {
                mean_re[k] = 0.0;
                mean_im[k] = 0.0;
            }
        }
        for (size_t k = 0; k < 3 * n_harm; ++k) {
            cyc_re[k] = 0.0;
            cyc_im[k] = 0.0;
        }

        if (last_time == 0.0) {
            /* Only possible when discard_time == 0, n_cycle == 1, n_phase == 1. */
            const double *cw = cos_w;
            const double *sw = sin_w;
            for (int j = 0; j < 3; ++j) {
                for (size_t m = 0; m < n_harm; ++m) {
                    size_t idx = (size_t)j * n_harm + m;
                    cyc_re[idx] = y0[j] * cw[m];
                    cyc_im[idx] = y0[j] * sw[m];
                }
            }
            size_t base = cond * 3 * n_harm;
            for (size_t k = 0; k < 3 * n_harm; ++k) {
                out_real[base + k] = cyc_re[k];
                out_imag[base + k] = cyc_im[k];
            }
            if (out_moments) {
                size_t mbase = cond * 10;
                double xy = y0[0] * y0[1];
                int pos = y0[0] > 0.0;
                out_moments[mbase + 0] = xy;
                out_moments[mbase + 1] = 0.0;
                out_moments[mbase + 2] = pos ? 1.0 : 0.0;
                out_moments[mbase + 3] = pos ? y0[0] : 0.0;
                out_moments[mbase + 4] = pos ? 0.0 : y0[0];
                out_moments[mbase + 5] = pos ? y0[2] : 0.0;
                out_moments[mbase + 6] = pos ? 0.0 : y0[2];
                out_moments[mbase + 7] = pos ? xy : 0.0;
                out_moments[mbase + 8] = y0[0] * y0[0];
                out_moments[mbase + 9] = y0[2] * y0[2];
            }
            continue;
        }

        double y[3] = {y0[0], y0[1], y0[2]};
        double f[3];
        lorenz_rhs(0.0, y, &params, f);
        double t = 0.0;
        double h_abs = (first_step > 0.0)
            ? first_step
            : select_initial_step(&params, 0.0, y, last_time, max_step, f, rtol, atol);

        double K[16][3];
        double F[7][3];
        double y_old[3];
        double h_prev = 0.0;

        size_t c_idx = 0;
        size_t q_idx = 0;
        size_t s_idx = 0;
        double t_next = discard_time;
        double sum_xy = 0.0, sum_x2 = 0.0, sum_z2 = 0.0;
        double sum_p_pos = 0.0, sum_x_pos = 0.0, sum_x_neg = 0.0;
        double sum_z_pos = 0.0, sum_z_neg = 0.0, sum_xy_pos = 0.0;
        double z_first = 0.0, z_last = 0.0;

        while (t < last_time && s_idx < total_samples) {
            double t_old = t;
            int rc = dop853_step(
                &params, &t, y, f, &h_abs, last_time, max_step,
                rtol, atol, K, &h_prev, y_old
            );
            if (rc != 0) {
                status = rc;
                goto cleanup;
            }

            if (s_idx < total_samples && t_next <= t) {
                dop853_build_dense(&params, t_old, h_prev, y_old, y, K, F);
                while (s_idx < total_samples && t_next <= t) {
                    double val[3];
                    dop853_eval_dense(t_next, t_old, h_prev, y_old, F, val);
                    if (!isfinite(val[0]) || !isfinite(val[1]) || !isfinite(val[2])) {
                        status = -2;
                        goto cleanup;
                    }

                    if (out_moments) {
                        double vx = val[0], vy = val[1], vz = val[2];
                        double xy = vx * vy;
                        sum_xy += xy;
                        sum_x2 += vx * vx;
                        sum_z2 += vz * vz;
                        if (vx > 0.0) {
                            sum_p_pos += 1.0;
                            sum_x_pos += vx;
                            sum_z_pos += vz;
                            sum_xy_pos += xy;
                        } else {
                            sum_x_neg += vx;
                            sum_z_neg += vz;
                        }
                        if (s_idx == 0) {
                            z_first = vz;
                        }
                        if (s_idx + 1 == total_samples) {
                            z_last = vz;
                        }
                    }

                    const double *cw = cos_w + q_idx * n_harm;
                    const double *sw = sin_w + q_idx * n_harm;
                    for (int j = 0; j < 3; ++j) {
                        double vj = val[j];
                        size_t row = (size_t)j * n_harm;
                        for (size_t m = 0; m < n_harm; ++m) {
                            cyc_re[row + m] += vj * cw[m];
                            cyc_im[row + m] += vj * sw[m];
                        }
                    }

                    ++s_idx;
                    ++q_idx;
                    if (q_idx == n_phase) {
                        if (average_cycles) {
                            for (size_t k = 0; k < 3 * n_harm; ++k) {
                                mean_re[k] += cyc_re[k] * inv_n_phase;
                                mean_im[k] += cyc_im[k] * inv_n_phase;
                                cyc_re[k] = 0.0;
                                cyc_im[k] = 0.0;
                            }
                        } else {
                            /* Output shape (n_cond, 3, n_cycle, n_harm). */
                            for (int j = 0; j < 3; ++j) {
                                size_t dst = ((cond * 3 + (size_t)j) * n_cycle + c_idx) * n_harm;
                                size_t src = (size_t)j * n_harm;
                                for (size_t m = 0; m < n_harm; ++m) {
                                    out_real[dst + m] = cyc_re[src + m] * inv_n_phase;
                                    out_imag[dst + m] = cyc_im[src + m] * inv_n_phase;
                                    cyc_re[src + m] = 0.0;
                                    cyc_im[src + m] = 0.0;
                                }
                            }
                        }
                        q_idx = 0;
                        ++c_idx;
                    }
                    if (s_idx < total_samples) {
                        t_next = discard_time + ((double)c_idx + (double)q_idx / (double)n_phase) * period;
                    }
                }
            }
        }
        if (s_idx != total_samples) {
            status = -3;
            goto cleanup;
        }
        if (average_cycles) {
            size_t base = cond * 3 * n_harm;
            for (size_t k = 0; k < 3 * n_harm; ++k) {
                out_real[base + k] = mean_re[k] * inv_n_cycle;
                out_imag[base + k] = mean_im[k] * inv_n_cycle;
            }
        }
        if (out_moments) {
            double inv_total = 1.0 / (double)total_samples;
            double total_t = (double)n_cycle * period;
            size_t mbase = cond * 10;
            out_moments[mbase + 0] = sum_xy * inv_total;
            out_moments[mbase + 1] = (z_last - z_first) / total_t;
            out_moments[mbase + 2] = sum_p_pos * inv_total;
            out_moments[mbase + 3] = sum_x_pos * inv_total;
            out_moments[mbase + 4] = sum_x_neg * inv_total;
            out_moments[mbase + 5] = sum_z_pos * inv_total;
            out_moments[mbase + 6] = sum_z_neg * inv_total;
            out_moments[mbase + 7] = sum_xy_pos * inv_total;
            out_moments[mbase + 8] = sum_x2 * inv_total;
            out_moments[mbase + 9] = sum_z2 * inv_total;
        }
    }

cleanup:
    free(cos_w);
    free(sin_w);
    free(cyc_re);
    free(cyc_im);
    free(mean_re);
    free(mean_im);
    return status;
}

int lorenz_dop853_fourier_block(
    const double y0[3],
    const double *forcing_vectors,
    const double *forcing_phases,
    int has_forcing_phases,
    size_t n_cond,
    double omega,
    double phase,
    double discard_time,
    size_t n_cycle,
    size_t n_phase,
    const long long *harmonics,
    size_t n_harm,
    double sigma,
    double rho,
    double beta,
    double rtol,
    const double atol[3],
    double max_step,
    double first_step,
    int average_cycles,
    double *out_real,
    double *out_imag
) {
    return lorenz_dop853_fourier_block_impl(
        y0, forcing_vectors, forcing_phases, has_forcing_phases, n_cond,
        omega, phase, discard_time, n_cycle, n_phase, harmonics, n_harm,
        sigma, rho, beta, rtol, atol, max_step, first_step, average_cycles,
        out_real, out_imag, NULL
    );
}

int lorenz_dop853_dc_moments_block(
    const double y0[3],
    const double *forcing_vectors,
    const double *forcing_phases,
    int has_forcing_phases,
    size_t n_cond,
    double omega,
    double phase,
    double discard_time,
    size_t n_cycle,
    size_t n_phase,
    const long long *harmonics,
    size_t n_harm,
    double sigma,
    double rho,
    double beta,
    double rtol,
    const double atol[3],
    double max_step,
    double first_step,
    double *out_real,
    double *out_imag,
    double *out_moments
) {
    if (!out_moments) {
        return -3;
    }
    return lorenz_dop853_fourier_block_impl(
        y0, forcing_vectors, forcing_phases, has_forcing_phases, n_cond,
        omega, phase, discard_time, n_cycle, n_phase, harmonics, n_harm,
        sigma, rho, beta, rtol, atol, max_step, first_step, 1,
        out_real, out_imag, out_moments
    );
}

/* =========================================================================
 * Dual-frequency forced Lorenz-63 DOP853 kernel with arbitrary frequency
 * projection via phase-rotation recurrence (zero-aliasing on integer
 * superperiods). Single-frequency hot path above is untouched.
 * ========================================================================= */

typedef struct {
    double sigma;
    double rho;
    double beta;
    double omega1;
    double omega2;
    double a1[3];
    double phi1[3];
    double a2[3];
    double phi2[3];
    int has_f1;
    int has_f2;
} LorenzDualParams;

static inline void lorenz_dual_rhs(
    double t,
    const double y[3],
    const LorenzDualParams *p,
    double out[3]
) {
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (p->has_f1) {
        double w1t = p->omega1 * t;
        if (p->a1[0] != 0.0) fx += p->a1[0] * sin(w1t + p->phi1[0]);
        if (p->a1[1] != 0.0) fy += p->a1[1] * sin(w1t + p->phi1[1]);
        if (p->a1[2] != 0.0) fz += p->a1[2] * sin(w1t + p->phi1[2]);
    }
    if (p->has_f2) {
        double w2t = p->omega2 * t;
        if (p->a2[0] != 0.0) fx += p->a2[0] * sin(w2t + p->phi2[0]);
        if (p->a2[1] != 0.0) fy += p->a2[1] * sin(w2t + p->phi2[1]);
        if (p->a2[2] != 0.0) fz += p->a2[2] * sin(w2t + p->phi2[2]);
    }
    out[0] = p->sigma * (y[1] - y[0]) + fx;
    out[1] = p->rho * y[0] - y[1] - y[0] * y[2] + fy;
    out[2] = y[0] * y[1] - p->beta * y[2] + fz;
}

static double select_initial_step_dual(
    const LorenzDualParams *p,
    double t0,
    const double y0[3],
    double t_bound,
    double max_step,
    const double f0[3],
    double rtol,
    const double atol[3]
) {
    double interval_length = fabs(t_bound - t0);
    if (interval_length == 0.0) {
        return 0.0;
    }

    double scale[3];
    double d0_sq = 0.0, d1_sq = 0.0;
    for (int j = 0; j < 3; ++j) {
        scale[j] = atol[j] + fabs(y0[j]) * rtol;
        double r0 = y0[j] / scale[j];
        double r1 = f0[j] / scale[j];
        d0_sq += r0 * r0;
        d1_sq += r1 * r1;
    }
    double d0 = sqrt(d0_sq / 3.0);
    double d1 = sqrt(d1_sq / 3.0);

    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : (0.01 * d0 / d1);
    if (h0 > interval_length) {
        h0 = interval_length;
    }

    double y1[3], f1[3];
    for (int j = 0; j < 3; ++j) {
        y1[j] = y0[j] + h0 * f0[j];
    }
    lorenz_dual_rhs(t0 + h0, y1, p, f1);

    double d2_sq = 0.0;
    for (int j = 0; j < 3; ++j) {
        double r2 = (f1[j] - f0[j]) / scale[j];
        d2_sq += r2 * r2;
    }
    double d2 = sqrt(d2_sq / 3.0) / h0;

    double h1;
    if (d1 <= 1e-15 && d2 <= 1e-15) {
        double cand = h0 * 1e-3;
        h1 = (1e-6 > cand) ? 1e-6 : cand;
    } else {
        double m = (d1 > d2) ? d1 : d2;
        h1 = pow(0.01 / m, 0.125);
    }

    double res = 100.0 * h0;
    if (h1 < res) res = h1;
    if (interval_length < res) res = interval_length;
    if (max_step < res) res = max_step;
    return res;
}

static int dop853_step_dual(
    const LorenzDualParams *p,
    double *t_ptr,
    double y[3],
    double f[3],
    double *h_abs_ptr,
    double t_bound,
    double max_step,
    double rtol,
    const double atol[3],
    double K[16][3],
    double *h_prev_ptr,
    double y_old[3]
) {
    double t = *t_ptr;
    double min_step = 10.0 * fabs(nextafter(t, INFINITY) - t);
    double h_abs = *h_abs_ptr;
    if (h_abs > max_step) {
        h_abs = max_step;
    } else if (h_abs < min_step) {
        h_abs = min_step;
    }

    int step_accepted = 0;
    int step_rejected = 0;
    double t_new = t;
    double h = h_abs;
    double y_new[3];
    double f_new[3];
    double y_tmp[3];

    while (!step_accepted) {
        if (h_abs < min_step) {
            return -1;
        }

        h = h_abs;
        t_new = t + h;
        if (t_new > t_bound) {
            t_new = t_bound;
        }
        h = t_new - t;
        h_abs = fabs(h);

        for (int j = 0; j < 3; ++j) {
            K[0][j] = f[j];
        }
        for (int s = 1; s < 12; ++s) {
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int m = 0; m < s; ++m) {
                    acc += DOP_A[s][m] * K[m][j];
                }
                y_tmp[j] = y[j] + acc * h;
            }
            lorenz_dual_rhs(t + DOP_C[s] * h, y_tmp, p, K[s]);
        }

        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int m = 0; m < 12; ++m) {
                acc += DOP_B[m] * K[m][j];
            }
            y_new[j] = y[j] + h * acc;
        }
        lorenz_dual_rhs(t + h, y_new, p, f_new);
        for (int j = 0; j < 3; ++j) {
            K[12][j] = f_new[j];
        }

        if (!isfinite(y_new[0]) || !isfinite(y_new[1]) || !isfinite(y_new[2]) ||
            !isfinite(f_new[0]) || !isfinite(f_new[1]) || !isfinite(f_new[2])) {
            return -2;
        }

        double err5_norm_2 = 0.0;
        double err3_norm_2 = 0.0;
        for (int j = 0; j < 3; ++j) {
            double scale = atol[j] + fmax(fabs(y[j]), fabs(y_new[j])) * rtol;
            double e5 = 0.0;
            double e3 = 0.0;
            for (int m = 0; m < 13; ++m) {
                e5 += K[m][j] * DOP_E5[m];
                e3 += K[m][j] * DOP_E3[m];
            }
            e5 /= scale;
            e3 /= scale;
            err5_norm_2 += e5 * e5;
            err3_norm_2 += e3 * e3;
        }

        double error_norm = 0.0;
        if (err5_norm_2 != 0.0 || err3_norm_2 != 0.0) {
            double denom = err5_norm_2 + 0.01 * err3_norm_2;
            error_norm = fabs(h) * err5_norm_2 / sqrt(denom * 3.0);
        }
        if (!isfinite(error_norm)) {
            return -2;
        }

        if (error_norm < 1.0) {
            double factor;
            if (error_norm == 0.0) {
                factor = DOP853_MAX_FACTOR;
            } else {
                factor = DOP853_SAFETY * pow(error_norm, DOP853_ERROR_EXPONENT);
                if (factor > DOP853_MAX_FACTOR) {
                    factor = DOP853_MAX_FACTOR;
                }
            }
            if (step_rejected && factor > 1.0) {
                factor = 1.0;
            }
            h_abs *= factor;
            step_accepted = 1;
        } else {
            double factor = DOP853_SAFETY * pow(error_norm, DOP853_ERROR_EXPONENT);
            if (factor < DOP853_MIN_FACTOR) {
                factor = DOP853_MIN_FACTOR;
            }
            h_abs *= factor;
            step_rejected = 1;
        }
    }

    *h_prev_ptr = h;
    for (int j = 0; j < 3; ++j) {
        y_old[j] = y[j];
        y[j] = y_new[j];
        f[j] = f_new[j];
    }
    *t_ptr = t_new;
    *h_abs_ptr = h_abs;
    return 0;
}

static void dop853_build_dense_dual(
    const LorenzDualParams *p,
    double t_old,
    double h,
    const double y_old[3],
    const double y_new[3],
    double K[16][3],
    double F[7][3]
) {
    double y_tmp[3];
    for (int s = 13; s < 16; ++s) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int m = 0; m < s; ++m) {
                acc += K[m][j] * DOP_A[s][m];
            }
            y_tmp[j] = y_old[j] + acc * h;
        }
        lorenz_dual_rhs(t_old + DOP_C[s] * h, y_tmp, p, K[s]);
    }

    for (int j = 0; j < 3; ++j) {
        double dy = y_new[j] - y_old[j];
        F[0][j] = dy;
        F[1][j] = h * K[0][j] - dy;
        F[2][j] = 2.0 * dy - h * (K[12][j] + K[0][j]);
        for (int r = 0; r < 4; ++r) {
            double acc = 0.0;
            for (int m = 0; m < 16; ++m) {
                acc += DOP_D[r][m] * K[m][j];
            }
            F[3 + r][j] = h * acc;
        }
    }
}

int lorenz_dop853_dual_freq_block(
    const double y0[3],
    const double *forcing1_vectors,
    const double *forcing1_phases,
    int has_forcing1_phases,
    double omega1,
    const double *forcing2_vectors,
    const double *forcing2_phases,
    int has_forcing2_phases,
    double omega2,
    size_t n_cond,
    double discard_time,
    double dt,
    size_t n_samples,
    const double *target_omegas,
    size_t n_freq,
    double sigma,
    double rho,
    double beta,
    double rtol,
    const double atol[3],
    double max_step,
    double first_step,
    double *out_real,
    double *out_imag,
    double *out_moments
) {
    if (!y0 || !forcing1_vectors || !forcing2_vectors || !target_omegas || !atol ||
        !out_real || !out_imag || n_cond == 0 || n_samples == 0 || n_freq == 0 ||
        omega1 < 0.0 || omega2 < 0.0 || !isfinite(omega1) || !isfinite(omega2) ||
        discard_time < 0.0 || !isfinite(discard_time) || dt <= 0.0 || !isfinite(dt) ||
        rtol <= 0.0 || max_step <= 0.0) {
        return -3;
    }
    if ((has_forcing1_phases && !forcing1_phases) || (has_forcing2_phases && !forcing2_phases)) {
        return -3;
    }
    for (size_t c = 0; c < n_cond * 3; ++c) {
        if (!isfinite(forcing1_vectors[c]) || !isfinite(forcing2_vectors[c])) {
            return -3;
        }
        if (has_forcing1_phases && !isfinite(forcing1_phases[c])) {
            return -3;
        }
        if (has_forcing2_phases && !isfinite(forcing2_phases[c])) {
            return -3;
        }
    }
    if (!isfinite(y0[0]) || !isfinite(y0[1]) || !isfinite(y0[2])) {
        return -2;
    }
    for (size_t m = 0; m < n_freq; ++m) {
        if (!isfinite(target_omegas[m])) {
            return -3;
        }
    }

    double last_time = discard_time + (double)(n_samples - 1) * dt;
    if (!isfinite(last_time)) {
        return -3;
    }

    double *step_re = (double *)malloc(n_freq * sizeof(double));
    double *step_im = (double *)malloc(n_freq * sizeof(double));
    double *ph_re = (double *)malloc(n_freq * sizeof(double));
    double *ph_im = (double *)malloc(n_freq * sizeof(double));
    double *acc_re = (double *)malloc(3 * n_freq * sizeof(double));
    double *acc_im = (double *)malloc(3 * n_freq * sizeof(double));
    if (!step_re || !step_im || !ph_re || !ph_im || !acc_re || !acc_im) {
        free(step_re);
        free(step_im);
        free(ph_re);
        free(ph_im);
        free(acc_re);
        free(acc_im);
        return -3;
    }

    for (size_t m = 0; m < n_freq; ++m) {
        double ang = target_omegas[m] * dt;
        step_re[m] = cos(ang);
        step_im[m] = -sin(ang);
    }

    double inv_n_samples = 1.0 / (double)n_samples;
    double total_t = (double)n_samples * dt;
    int status = 0;

    for (size_t cond = 0; cond < n_cond; ++cond) {
        const double *fv1 = forcing1_vectors + cond * 3;
        const double *fp1 = has_forcing1_phases ? (forcing1_phases + cond * 3) : NULL;
        const double *fv2 = forcing2_vectors + cond * 3;
        const double *fp2 = has_forcing2_phases ? (forcing2_phases + cond * 3) : NULL;

        LorenzDualParams params = {
            sigma, rho, beta, omega1, omega2,
            {fv1[0], fv1[1], fv1[2]},
            {fp1 ? fp1[0] : 0.0, fp1 ? fp1[1] : 0.0, fp1 ? fp1[2] : 0.0},
            {fv2[0], fv2[1], fv2[2]},
            {fp2 ? fp2[0] : 0.0, fp2 ? fp2[1] : 0.0, fp2 ? fp2[2] : 0.0},
            (fv1[0] != 0.0 || fv1[1] != 0.0 || fv1[2] != 0.0),
            (fv2[0] != 0.0 || fv2[1] != 0.0 || fv2[2] != 0.0)
        };

        for (size_t k = 0; k < 3 * n_freq; ++k) {
            acc_re[k] = 0.0;
            acc_im[k] = 0.0;
        }

        if (last_time == 0.0) {
            for (int j = 0; j < 3; ++j) {
                for (size_t m = 0; m < n_freq; ++m) {
                    size_t idx = (size_t)j * n_freq + m;
                    acc_re[idx] = y0[j];
                    acc_im[idx] = 0.0;
                }
            }
            size_t base = cond * 3 * n_freq;
            for (size_t k = 0; k < 3 * n_freq; ++k) {
                out_real[base + k] = acc_re[k];
                out_imag[base + k] = acc_im[k];
            }
            if (out_moments) {
                size_t mbase = cond * 10;
                double xy = y0[0] * y0[1];
                int pos = y0[0] > 0.0;
                out_moments[mbase + 0] = xy;
                out_moments[mbase + 1] = 0.0;
                out_moments[mbase + 2] = pos ? 1.0 : 0.0;
                out_moments[mbase + 3] = pos ? y0[0] : 0.0;
                out_moments[mbase + 4] = pos ? 0.0 : y0[0];
                out_moments[mbase + 5] = pos ? y0[2] : 0.0;
                out_moments[mbase + 6] = pos ? 0.0 : y0[2];
                out_moments[mbase + 7] = pos ? xy : 0.0;
                out_moments[mbase + 8] = y0[0] * y0[0];
                out_moments[mbase + 9] = y0[2] * y0[2];
            }
            continue;
        }

        double y[3] = {y0[0], y0[1], y0[2]};
        double f[3];
        lorenz_dual_rhs(0.0, y, &params, f);
        double t = 0.0;
        double h_abs = (first_step > 0.0)
            ? first_step
            : select_initial_step_dual(&params, 0.0, y, last_time, max_step, f, rtol, atol);

        double K[16][3];
        double F[7][3];
        double y_old[3];
        double h_prev = 0.0;

        size_t s_idx = 0;
        double t_next = discard_time;
        double sum_xy = 0.0, sum_x2 = 0.0, sum_z2 = 0.0;
        double sum_p_pos = 0.0, sum_x_pos = 0.0, sum_x_neg = 0.0;
        double sum_z_pos = 0.0, sum_z_neg = 0.0, sum_xy_pos = 0.0;
        double z_first = 0.0, z_last = 0.0;

        while (t < last_time && s_idx < n_samples) {
            double t_old = t;
            int rc = dop853_step_dual(
                &params, &t, y, f, &h_abs, last_time, max_step,
                rtol, atol, K, &h_prev, y_old
            );
            if (rc != 0) {
                status = rc;
                goto cleanup;
            }

            if (s_idx < n_samples && t_next <= t) {
                dop853_build_dense_dual(&params, t_old, h_prev, y_old, y, K, F);
                while (s_idx < n_samples && t_next <= t) {
                    double val[3];
                    dop853_eval_dense(t_next, t_old, h_prev, y_old, F, val);
                    if (!isfinite(val[0]) || !isfinite(val[1]) || !isfinite(val[2])) {
                        status = -2;
                        goto cleanup;
                    }

                    if (out_moments) {
                        double vx = val[0], vy = val[1], vz = val[2];
                        double xy = vx * vy;
                        sum_xy += xy;
                        sum_x2 += vx * vx;
                        sum_z2 += vz * vz;
                        if (vx > 0.0) {
                            sum_p_pos += 1.0;
                            sum_x_pos += vx;
                            sum_z_pos += vz;
                            sum_xy_pos += xy;
                        } else {
                            sum_x_neg += vx;
                            sum_z_neg += vz;
                        }
                        if (s_idx == 0) {
                            z_first = vz;
                        }
                        if (s_idx + 1 == n_samples) {
                            z_last = vz;
                        }
                    }

                    /* Re-anchor phasors every 256 samples to eliminate recurrence drift. */
                    if ((s_idx & 255U) == 0U) {
                        for (size_t m = 0; m < n_freq; ++m) {
                            double ang = target_omegas[m] * t_next;
                            ph_re[m] = cos(ang);
                            ph_im[m] = -sin(ang);
                        }
                    }

                    for (int j = 0; j < 3; ++j) {
                        double vj = val[j];
                        size_t row = (size_t)j * n_freq;
                        for (size_t m = 0; m < n_freq; ++m) {
                            acc_re[row + m] += vj * ph_re[m];
                            acc_im[row + m] += vj * ph_im[m];
                        }
                    }

                    for (size_t m = 0; m < n_freq; ++m) {
                        double nr = ph_re[m] * step_re[m] - ph_im[m] * step_im[m];
                        double ni = ph_re[m] * step_im[m] + ph_im[m] * step_re[m];
                        ph_re[m] = nr;
                        ph_im[m] = ni;
                    }

                    ++s_idx;
                    if (s_idx < n_samples) {
                        t_next = discard_time + (double)s_idx * dt;
                    }
                }
            }
        }
        if (s_idx != n_samples) {
            status = -3;
            goto cleanup;
        }

        size_t base = cond * 3 * n_freq;
        for (size_t k = 0; k < 3 * n_freq; ++k) {
            out_real[base + k] = acc_re[k] * inv_n_samples;
            out_imag[base + k] = acc_im[k] * inv_n_samples;
        }
        if (out_moments) {
            size_t mbase = cond * 10;
            out_moments[mbase + 0] = sum_xy * inv_n_samples;
            out_moments[mbase + 1] = (z_last - z_first) / total_t;
            out_moments[mbase + 2] = sum_p_pos * inv_n_samples;
            out_moments[mbase + 3] = sum_x_pos * inv_n_samples;
            out_moments[mbase + 4] = sum_x_neg * inv_n_samples;
            out_moments[mbase + 5] = sum_z_pos * inv_n_samples;
            out_moments[mbase + 6] = sum_z_neg * inv_n_samples;
            out_moments[mbase + 7] = sum_xy_pos * inv_n_samples;
            out_moments[mbase + 8] = sum_x2 * inv_n_samples;
            out_moments[mbase + 9] = sum_z2 * inv_n_samples;
        }
    }

cleanup:
    free(step_re);
    free(step_im);
    free(ph_re);
    free(ph_im);
    free(acc_re);
    free(acc_im);
    return status;
}
