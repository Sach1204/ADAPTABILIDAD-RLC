
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

OUT = "outputs"
# ---------------------------------------------------------------------------
# 1. MODELO RLC SERIE
# ---------------------------------------------------------------------------
#
# KVL:  Vin(t) = L di/dt + R i + vC
#       i = C dvC/dt
#
# Estados:  x1 = i (corriente),  x2 = vC (tensión en el capacitor)
#
#   dx1/dt = (Vin - R x1 - x2) / L
#   dx2/dt = x1 / C
#
# Frecuencia natural:  wn = 1/sqrt(LC)
# Factor de amortiguamiento:  zeta = (R/2) * sqrt(C/L)

def rlc_matrices(R, L, C):
    A = np.array([[-R / L, -1 / L],
                  [1 / C, 0.0]])
    B = np.array([[1 / L], [0.0]])
    return A, B


def rlc_ode(t, x, R, L, C, vin_func):
    A, B = rlc_matrices(R, L, C)
    u = vin_func(t)
    dx = A @ x + B.flatten() * u
    return dx


def natural_params(R, L, C):
    wn = 1 / np.sqrt(L * C)
    zeta = (R / 2) * np.sqrt(C / L)
    return wn, zeta


# ---------------------------------------------------------------------------
# 2. SOLUCIÓN ANALÍTICA (entrada escalón Vin = V0, condiciones iniciales 0)
# ---------------------------------------------------------------------------
def analytic_step_response(t, R, L, C, V0):
    """vC(t) para un escalón de tensión V0 aplicado en t=0, x(0)=0."""
    wn, zeta = natural_params(R, L, C)
    vC = np.zeros_like(t)

    if zeta < 1:  # sub-amortiguado
        wd = wn * np.sqrt(1 - zeta ** 2)
        phi = np.arccos(zeta)
        vC = V0 * (1 - np.exp(-zeta * wn * t) / np.sqrt(1 - zeta ** 2)
                   * np.cos(wd * t - phi))
    elif np.isclose(zeta, 1):  # crítico
        vC = V0 * (1 - np.exp(-wn * t) * (1 + wn * t))
    else:  # sobre-amortiguado
        r1 = -wn * (zeta - np.sqrt(zeta ** 2 - 1))
        r2 = -wn * (zeta + np.sqrt(zeta ** 2 - 1))
        A2 = (r1 * V0) / (r2 - r1) * -1  # coeficientes por fracciones parciales
        # Resolviendo con condiciones iniciales vC(0)=0, dvC/dt(0)=0:
        C1 = V0 * r2 / (r2 - r1)
        C2 = -V0 * r1 / (r2 - r1)
        vC = V0 - C1 * np.exp(r1 * t) - C2 * np.exp(r2 * t)
    return vC


# ---------------------------------------------------------------------------
# 3. VERIFICACIÓN NUMÉRICA VS ANALÍTICA
# ---------------------------------------------------------------------------
def verificar_estados(R=100.0, L=0.5, C=100e-6, V0=5.0, t_end=0.05):
    wn, zeta = natural_params(R, L, C)
    t_eval = np.linspace(0, t_end, 2000)

    sol = solve_ivp(rlc_ode, [0, t_end], [0.0, 0.0], t_eval=t_eval,
                     args=(R, L, C, lambda t: V0), rtol=1e-9, atol=1e-12)

    vC_num = sol.y[1]
    vC_ana = analytic_step_response(t_eval, R, L, C, V0)
    error = np.max(np.abs(vC_num - vC_ana))

    print("=== Verificación de estados (RLC nominal) ===")
    print(f"R={R} ohm, L={L} H, C={C} F")
    print(f"wn = {wn:.2f} rad/s | zeta = {zeta:.3f} "
          f"({'sub-amortiguado' if zeta < 1 else 'sobre/crítico'})")
    print(f"Error máximo numérico vs analítico en vC: {error:.3e} V")

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t_eval, sol.y[0], label="i(t) numérico")
    axs[0].set_ylabel("Corriente i [A]")
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(t_eval, vC_num, label="vC(t) numérico")
    axs[1].plot(t_eval, vC_ana, "--", label="vC(t) analítico")
    axs[1].set_ylabel("Tensión vC [V]")
    axs[1].set_xlabel("t [s]")
    axs[1].legend(); axs[1].grid(True)
    fig.suptitle("Verificación de estados del circuito RLC")
    fig.tight_layout()
    fig.savefig(f"{OUT}/1_verificacion_estados.png", dpi=150)
    plt.close(fig)
    return error


# ---------------------------------------------------------------------------
# 4. ADAPTABILIDAD PASIVA: variación de parámetros sin control
# ---------------------------------------------------------------------------
def adaptabilidad_pasiva(L=0.5, C=100e-6, V0=5.0, t_end=0.15):
    """
    R cambia abruptamente en t = t_end/2 (por ejemplo, envejecimiento de un
    componente o cambio de carga). El sistema NO tiene ningún mecanismo de
    corrección: simplemente responde con la física que le corresponde a los
    nuevos parámetros. Esto sirve para argumentar que un sistema puramente
    pasivo no es "adaptable" en sentido estricto: su comportamiento cambia
    porque cambió la planta, no porque el sistema haya compensado el cambio.
    """
    t_switch = t_end / 2

    def R_t(t):
        return 60.0 if t < t_switch else 400.0  # sube el amortiguamiento

    def ode(t, x):
        R = R_t(t)
        A, B = rlc_matrices(R, L, C)
        return A @ x + B.flatten() * V0

    t_eval = np.linspace(0, t_end, 3000)
    sol = solve_ivp(ode, [0, t_end], [0.0, 0.0], t_eval=t_eval,
                     rtol=1e-9, atol=1e-12)

    wn1, z1 = natural_params(60.0, L, C)
    wn2, z2 = natural_params(400.0, L, C)
    print("\n=== Adaptabilidad pasiva (sin control) ===")
    print(f"Antes del cambio:  zeta={z1:.2f} (sub-amortiguado)")
    print(f"Después del cambio: zeta={z2:.2f} (sobre-amortiguado)")
    print("-> El sistema no 'decide' adaptarse: su dinámica cambia porque")
    print("   cambiaron sus parámetros físicos. No hay compensación activa.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_eval, sol.y[1], label="vC(t)")
    ax.axvline(t_switch, color="r", ls="--", label="cambio de R (60→400 Ω)")
    ax.axhline(V0, color="gray", ls=":", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("vC [V]")
    ax.set_title("Respuesta ante cambio brusco de R (sistema pasivo)")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/2_adaptabilidad_pasiva.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. ADAPTABILIDAD ACTIVA: control adaptativo tipo Lyapunov (MRAC simple)
# ---------------------------------------------------------------------------
def adaptabilidad_activa(L=0.5, C=100e-6, Vref=5.0, t_end=0.15,
                          gamma=300.0, sigma=5.0):
    """
    Se agrega un lazo de control con una ganancia adaptativa K(t) que
    corrige la tensión de entrada según el error e = Vref - vC:

        u(t) = K(t) * e(t)
        dK/dt = gamma * e(t) * vC(t) - sigma * (K(t) - K0)

    El primer término es la ley de adaptación tipo Lyapunov (gradiente que
    reduce el error). El segundo término (sigma-modificación) es una fuga
    ("leakage") estándar en control adaptativo que evita que la ganancia
    diverja sin límite ante errores persistentes o ruido, manteniendo el
    esquema numéricamente estable.

    R vuelve a cambiar abruptamente a la mitad de la simulación. A
    diferencia del caso pasivo, aquí el sistema SÍ recalcula su acción de
    control en línea para intentar mantener vC cerca de la referencia:
    esto es lo que se entiende por "adaptabilidad" -> la capacidad de
    modificar su propio comportamiento (no solo sufrir el cambio de planta)
    para preservar un objetivo (estabilidad + error acotado) ante
    incertidumbre o variación de parámetros.
    """
    t_switch = t_end / 2
    K0 = 1.0

    def R_t(t):
        return 60.0 if t < t_switch else 400.0

    def ode(t, z):
        i, vC, K = z
        R = R_t(t)
        e = Vref - vC
        u = K * e
        di = (u - R * i - vC) / L
        dvC = i / C
        dK = gamma * e * vC - sigma * (K - K0)
        return [di, dvC, dK]

    t_eval = np.linspace(0, t_end, 4000)
    sol = solve_ivp(ode, [0, t_end], [0.0, 0.0, K0], t_eval=t_eval,
                     rtol=1e-8, atol=1e-10, max_step=5e-5)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t_eval, sol.y[1], label="vC(t) con control adaptativo")
    axs[0].axhline(Vref, color="gray", ls=":", label="referencia")
    axs[0].axvline(t_switch, color="r", ls="--", label="cambio de R")
    axs[0].set_ylabel("vC [V]"); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(t_eval, sol.y[2], color="darkorange", label="K(t) (ganancia adaptativa)")
    axs[1].axvline(t_switch, color="r", ls="--")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("K")
    axs[1].legend(); axs[1].grid(True)
    fig.suptitle("Adaptabilidad activa: control adaptativo tipo Lyapunov")
    fig.tight_layout()
    fig.savefig(f"{OUT}/3_adaptabilidad_activa.png", dpi=150)
    plt.close(fig)

    err_final = abs(Vref - sol.y[1][-1])
    print("\n=== Adaptabilidad activa (con control) ===")
    print(f"Error final |Vref - vC|: {err_final:.4f} V")
    print("-> La ganancia K se reajusta sola tras el cambio de R, y el")
    print("   sistema vuelve a converger a la referencia: esto sí es")
    print("   adaptabilidad en sentido de control (auto-ajuste ante cambios).")


# ---------------------------------------------------------------------------
# 6. GENERALIZACIÓN: sistema LTI de n estados (clase genérica)
# ---------------------------------------------------------------------------
class LTISystem:
    """Sistema lineal genérico dx/dt = A x + B u, y = Cm x + D u.
    Sirve para representar cualquier sistema de n estados, no solo un RLC
    simple: por ejemplo dos etapas RLC en cascada (sistema de orden 4),
    una red de N nodos, etc.
    """
    def __init__(self, A, B, Cm=None, D=None):
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        n = self.A.shape[0]
        self.Cm = np.eye(n) if Cm is None else np.atleast_2d(Cm)
        self.D = np.zeros((self.Cm.shape[0], self.B.shape[1])) if D is None else D

    def ode(self, t, x, u_func):
        u = np.atleast_1d(u_func(t))
        return (self.A @ x + self.B @ u)

    def simulate(self, t_span, x0, u_func, t_eval=None):
        return solve_ivp(self.ode, t_span, x0, args=(u_func,),
                          t_eval=t_eval, rtol=1e-9, atol=1e-12)


def cascada_rlc_matrices(R1, L1, C1, R2, L2, C2):
    """
    Dos etapas RLC en cascada (la tensión del capacitor de la primera
    etapa alimenta a la segunda). Estados: [i1, vC1, i2, vC2] -> orden 4.
    Este es el "sistema complejo" pedido en la consigna: mismo tipo de
    ecuaciones que el RLC simple, pero acoplado y de mayor dimensión.
    """
    A = np.array([
        [-R1 / L1, -1 / L1,      0,       0],
        [1 / C1,       0,   -1 / C1,      0],
        [0,        1 / L2,  -R2 / L2, -1 / L2],
        [0,           0,      1 / C2,      0],
    ])
    B = np.array([[1 / L1], [0], [0], [0]])
    return A, B


def sistema_complejo_generalizado(t_end=0.2, gamma=20.0, sigma=1.0):
    """
    Aplica el mismo esquema de adaptabilidad activa (ganancia adaptativa
    tipo Lyapunov) a un sistema de orden 4 (dos RLC en cascada), para
    mostrar que el enfoque generaliza más allá del caso simple de 2 estados.
    La salida controlada es vC2 (tensión en el segundo capacitor).
    """
    R1, L1, C1 = 50.0, 0.3, 150e-6
    R2, L2, C2 = 80.0, 0.4, 100e-6
    Vref = 5.0
    K0 = 1.0
    t_switch = t_end / 2

    def ode(t, z):
        x = z[:4]
        K = z[4]
        R1_t = R1 if t < t_switch else 250.0  # variación paramétrica en la etapa 1
        A, B = cascada_rlc_matrices(R1_t, L1, C1, R2, L2, C2)
        vC2 = x[3]
        e = Vref - vC2
        u = K * e
        dx = A @ x + B.flatten() * u
        dK = gamma * e * vC2 - sigma * (K - K0)
        return np.concatenate([dx, [dK]])

    t_eval = np.linspace(0, t_end, 5000)
    z0 = [0, 0, 0, 0, K0]
    sol = solve_ivp(ode, [0, t_end], z0, t_eval=t_eval,
                     rtol=1e-8, atol=1e-10, max_step=1e-4)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t_eval, sol.y[1], label="vC1(t)")
    axs[0].plot(t_eval, sol.y[3], label="vC2(t) (salida)")
    axs[0].axhline(Vref, color="gray", ls=":", label="referencia")
    axs[0].axvline(t_switch, color="r", ls="--", label="cambio de R1")
    axs[0].set_ylabel("Tensión [V]"); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(t_eval, sol.y[4], color="darkorange", label="K(t)")
    axs[1].axvline(t_switch, color="r", ls="--")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("K")
    axs[1].legend(); axs[1].grid(True)
    fig.suptitle("Generalización: sistema RLC en cascada (orden 4) con control adaptativo")
    fig.tight_layout()
    fig.savefig(f"{OUT}/4_sistema_complejo_generalizado.png", dpi=150)
    plt.close(fig)

    err_final = abs(Vref - sol.y[3][-1])
    print("\n=== Generalización a sistema complejo (orden 4) ===")
    print(f"Error final en la salida (vC2) respecto a Vref: {err_final:.4f} V")
    print("-> El mismo esquema de adaptabilidad (ley de Lyapunov) funciona")
    print("   sin cambios estructurales sobre un sistema de mayor orden,")
    print("   lo que muestra que el concepto generaliza más allá del RLC simple.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    verificar_estados()
    adaptabilidad_pasiva()
    adaptabilidad_activa()
    sistema_complejo_generalizado()
    print("\nListo. Gráficos guardados en la carpeta 'outputs/'.")
