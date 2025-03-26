import pytest
from motionsim import Lagrangian
import sympy as sp

def test_lagrangian_calculation_simple():
    """Test simple Lagrangian calculations (1D harmonic oscillator and free fall)"""
    # Test case 1: 1D harmonic oscillator
    T1 = "0.5 * m * x_dot**2"
    V1 = "0.5 * k * x**2"
    lagrangian1 = Lagrangian(T1, V1, "x")
    
    # Create symbols for verification
    m, k, t = sp.symbols('m k t')
    x = sp.Function('x')(t)
    x_dot = x.diff(t)
    expected_L1 = 0.5 * m * x_dot**2 - 0.5 * k * x**2
    
    # Check if expressions are equivalent (might have different forms)
    assert sp.simplify(lagrangian1.L - expected_L1) == 0
    
    # Test case 2: Free falling object
    T2 = "0.5 * m * y_dot**2"
    V2 = "m * g * y"
    lagrangian2 = Lagrangian(T2, V2, "y")
    
    # Create symbols for verification
    g = sp.Symbol('g')
    y = sp.Function('y')(t)
    y_dot = y.diff(t)
    expected_L2 = 0.5 * m * y_dot**2 - m * g * y
    
    # Check if expressions are equivalent
    assert sp.simplify(lagrangian2.L - expected_L2) == 0

def test_lagrangian_calculation_complex():
    """Test complex Lagrangian calculations (double pendulum and particle in EM field)"""
    # Test case 1: Double pendulum
    T1 = "0.5*m1*l1**2*theta1_dot**2 + 0.5*m2*(l1**2*theta1_dot**2 + l2**2*theta2_dot**2 + 2*l1*l2*theta1_dot*theta2_dot*cos(theta1-theta2))"
    V1 = "-(m1+m2)*g*l1*cos(theta1) - m2*g*l2*cos(theta2)"
    lagrangian1 = Lagrangian(T1, V1, "theta1 theta2")
    
    # Create symbols for verification
    m1, m2, l1, l2, g, t = sp.symbols('m1 m2 l1 l2 g t')
    theta1 = sp.Function('theta1')(t)
    theta2 = sp.Function('theta2')(t)
    theta1_dot = theta1.diff(t)
    theta2_dot = theta2.diff(t)
    
    expected_L1 = 0.5*m1*l1**2*theta1_dot**2 + 0.5*m2*(l1**2*theta1_dot**2 + l2**2*theta2_dot**2 + 
                   2*l1*l2*theta1_dot*theta2_dot*sp.cos(theta1-theta2)) - \
                  (-(m1+m2)*g*l1*sp.cos(theta1) - m2*g*l2*sp.cos(theta2))
    
    # Check if expressions are equivalent
    assert sp.simplify(lagrangian1.L - expected_L1) == 0
    
    # Test case 2: Particle in electromagnetic field
    T2 = "0.5 * m * (x_dot**2 + y_dot**2 + z_dot**2)"
    V2 = "q * (phi - x_dot*Ax - y_dot*Ay - z_dot*Az)"
    lagrangian2 = Lagrangian(T2, V2, "x y z")
    
    # Create symbols for verification
    m, q, phi, Ax, Ay, Az = sp.symbols('m q phi Ax Ay Az')
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    z = sp.Function('z')(t)
    x_dot = x.diff(t)
    y_dot = y.diff(t)
    z_dot = z.diff(t)
    
    expected_L2 = 0.5 * m * (x_dot**2 + y_dot**2 + z_dot**2) - q * (phi - x_dot*Ax - y_dot*Ay - z_dot*Az)
    
    # Check if expressions are equivalent
    assert sp.simplify(lagrangian2.L - expected_L2) == 0

def test_equations_of_motion_simple():
    """Test simple equations of motion (1D oscillator and projectile)"""
    # Test case 1: 1D harmonic oscillator
    T1 = "0.5 * m * x_dot**2"
    V1 = "0.5 * k * x**2"
    lagrangian1 = Lagrangian(T1, V1, "x")
    
    # Create symbols for verification
    m, k, t = sp.symbols('m k t')
    x = sp.Function('x')(t)
    x_dot = x.diff(t)
    x_ddot = x_dot.diff(t)
    
    # Expected equation: m*x''(t) + k*x(t) = 0
    expected_eom1 = sp.Eq(m * x_ddot + k * x, 0)
    
    # Get the actual equation from the Lagrangian
    eom1 = lagrangian1.equations_of_motion()[0]
    
    # Check if equations are equivalent after simplification
    assert sp.simplify(eom1.lhs - expected_eom1.lhs) == 0
    assert sp.simplify(eom1.rhs - expected_eom1.rhs) == 0
    
    # Test case 2: Projectile motion
    T2 = "0.5 * m * (x_dot**2 + y_dot**2)"
    V2 = "m * g * y"
    lagrangian2 = Lagrangian(T2, V2, "x y")
    
    # Create symbols for verification
    g = sp.Symbol('g')
    y = sp.Function('y')(t)
    y_dot = y.diff(t)
    y_ddot = y_dot.diff(t)
    
    x = sp.Function('x')(t)
    x_dot = x.diff(t)
    x_ddot = x_dot.diff(t)
    
    # Expected equations:
    # 1. m*x''(t) = 0
    # 2. m*y''(t) + m*g = 0
    expected_eom_x = sp.Eq(m * x_ddot, 0)
    expected_eom_y = sp.Eq(m * y_ddot + m * g, 0)
    
    eoms = lagrangian2.equations_of_motion()
    
    # Find which equation corresponds to which coordinate
    for eom in eoms:
        if str(eom).find('x(t)') != -1:
            eom_x = eom
        elif str(eom).find('y(t)') != -1:
            eom_y = eom
            
    # Check if equations are equivalent
    assert sp.simplify(eom_x.lhs - expected_eom_x.lhs) == 0
    assert sp.simplify(eom_x.rhs - expected_eom_x.rhs) == 0
    assert sp.simplify(eom_y.lhs - expected_eom_y.lhs) == 0
    assert sp.simplify(eom_y.rhs - expected_eom_y.rhs) == 0

def test_equations_of_motion_complex():
    """Test complex equations of motion (pendulum and coupled oscillators)"""
    # Test case 1: Simple pendulum
    T1 = "0.5 * m * l**2 * theta_dot**2"
    V1 = "m * g * l * (1 - cos(theta))"
    lagrangian1 = Lagrangian(T1, V1, "theta")
    
    # Create symbols for verification
    m, l, g, t = sp.symbols('m l g t')
    theta = sp.Function('theta')(t)
    theta_dot = theta.diff(t)
    theta_ddot = theta_dot.diff(t)
    
    # Expected equation: m*l^2*theta''(t) + m*g*l*sin(theta(t)) = 0
    expected_eom1 = sp.Eq(m * l**2 * theta_ddot + m * g * l * sp.sin(theta), 0)
    
    # Get the actual equation from the Lagrangian
    eom1 = lagrangian1.equations_of_motion()[0]
    
    # Check if the pendulum equation is equivalent
    assert sp.simplify(eom1.lhs - expected_eom1.lhs) == 0
    assert sp.simplify(eom1.rhs - expected_eom1.rhs) == 0

    # Test case 2: Coupled oscillators
    T2 = "0.5 * m1 * x1_dot**2 + 0.5 * m2 * x2_dot**2"
    V2 = "0.5 * k1 * x1**2 + 0.5 * k2 * (x2 - x1)**2"
    lagrangian2 = Lagrangian(T2, V2, "x1 x2")
    
    # Create symbols for verification
    m1, m2, k1, k2, t = sp.symbols('m1 m2 k1 k2 t')
    x1 = sp.Function('x1')(t)
    x2 = sp.Function('x2')(t)
    x1_dot = x1.diff(t)
    x1_ddot = x1_dot.diff(t)
    x2_dot = x2.diff(t)
    x2_ddot = x2_dot.diff(t)
    
    # Expected equations:
    # For x1: m1*x1''(t) + k1*x1(t) - k2*(x2(t) - x1(t)) = 0
    # For x2: m2*x2''(t) + k2*(x2(t) - x1(t)) = 0
    expected_eom_x1 = sp.Eq(m1 * x1_ddot + k1 * x1 - k2 * (x2 - x1), 0)
    expected_eom_x2 = sp.Eq(m2 * x2_ddot + k2 * (x2 - x1), 0)
    
    eoms = lagrangian2.equations_of_motion()
    
    eom_x1 = None
    eom_x2 = None
    for eq in eoms:
        # Check coefficients of the second derivatives in the left-hand side
        coeff_x1 = sp.simplify(eq.lhs.coeff(x1_ddot))
        coeff_x2 = sp.simplify(eq.lhs.coeff(x2_ddot))
        if coeff_x1 != 0:
            eom_x1 = eq
        if coeff_x2 != 0:
            eom_x2 = eq

    # Ensure that both equations are found
    assert eom_x1 is not None, "Equation for x1 not found."
    assert eom_x2 is not None, "Equation for x2 not found."
    
    # Check if equations are equivalent
    assert sp.simplify(eom_x1.lhs - expected_eom_x1.lhs) == 0
    assert sp.simplify(eom_x1.rhs - expected_eom_x1.rhs) == 0
    assert sp.simplify(eom_x2.lhs - expected_eom_x2.lhs) == 0
    assert sp.simplify(eom_x2.rhs - expected_eom_x2.rhs) == 0


def test_equations_of_motion_complex2():
    """Test Euler–Lagrange equations for coupled harmonic oscillators with cross coupling."""
    # Define Lagrangian components as strings
    T = "0.5 * m * (x_dot**2 + y_dot**2)"
    V = "0.5 * k * (x**2 + y**2) + lam * x * y"
    lagrangian = Lagrangian(T, V, "x y")
    
    # Define symbols and functions
    m, k, lam, t = sp.symbols('m k lam t')
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    x_dot = x.diff(t)
    x_ddot = x_dot.diff(t)
    y_dot = y.diff(t)
    y_ddot = y_dot.diff(t)
    
    # Expected Euler–Lagrange equations:
    # m * x''(t) + k * x(t) + lam * y(t) = 0
    # m * y''(t) + k * y(t) + lam * x(t) = 0
    expected_eom_x = sp.Eq(m * x_ddot + k * x + lam * y, 0)
    expected_eom_y = sp.Eq(m * y_ddot + k * y + lam * x, 0)
    
    eoms = lagrangian.equations_of_motion()
    
    eom_x = None
    eom_y = None
    # Identify equations by inspecting coefficients of second derivatives
    for eq in eoms:
        if eq.lhs.coeff(x_ddot) != 0:
            eom_x = eq
        if eq.lhs.coeff(y_ddot) != 0:
            eom_y = eq
            
    assert eom_x is not None, "Equation for x not found."
    assert eom_y is not None, "Equation for y not found."
    
    # Compare the obtained equations with the expected ones
    assert sp.simplify(eom_x.lhs - expected_eom_x.lhs) == 0
    assert sp.simplify(eom_x.rhs - expected_eom_x.rhs) == 0
    assert sp.simplify(eom_y.lhs - expected_eom_y.lhs) == 0
    assert sp.simplify(eom_y.rhs - expected_eom_y.rhs) == 0
