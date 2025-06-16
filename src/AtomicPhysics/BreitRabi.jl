"""
Python Version: 2014_01_01
Julia port: 2025_05_22

Breit Rabi equation

@author: ispielma
"""

module BreitRabi
    using QGas.AtomicPhysics.AtomicConstants: μ_B_Hz_Gauss, State
    """
    breit_rabi(B, m_F, s, ΔE, I, g_J, g_I)  

    Breit Rabi Equation
    B is magnetic field in gauss
    m_F is magnetic sublevel
    s is a sign selecting the ground or excited manifold
    ΔE is hyperfine splitting
    I is the nuclear spin
    g_J is Landè g-factor
    g_I is nuclear g-factor
    """
    function breit_rabi(B::Real, m_F::Real, s::Int, ΔE::Real, I::Real, g_J::Real, g_I::Real)
        coeff = (g_J - g_I) * μ_B_Hz_Gauss * B / ΔE

        inside = 1 + 4*m_F/(2*I+1)*coeff + coeff^2
        return -ΔE / (2*(2*I + 1)) + g_I * μ_B_Hz_Gauss * m_F * B + s * ΔE * sqrt(inside) / 2
    end

    """
    breit_rabi(B, m_F, s, state::State)

    A method of breit_rabi() that takes an abstracted atom as a parameter.
    """
    breit_rabi(B, m_F, s, state::State) = breit_rabi(B, m_F, s, 2*state.A, state.i, state.g_J, state.g_I)

end