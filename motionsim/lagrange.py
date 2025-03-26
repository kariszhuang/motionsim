import sympy as sp
from typing import List, Union, Optional, Dict, Set

class Lagrangian:
    """
    Given kinetic energy T and potential energy V expressions, computes the Lagrangian L = T - V
    and derives the equations of motion using the Euler-Lagrange equations.
    """
    def __init__(self, T: str, V: str, coordinates: Union[str, List[str], None] = None) -> None:
        # Parse expressions
        T_static: sp.Expr = sp.sympify(T)
        V_static: sp.Expr = sp.sympify(V)
        
        # Compute the Lagrangian
        L_static: sp.Expr = T_static - V_static
        
        # Set up time symbol
        self.t: sp.Symbol = sp.symbols('t')
        
        # Parse coordinates
        self.coordinates: List[str] = self._parse_coordinates(T, V, coordinates)
        
        # Set up dynamic expressions
        self.T: sp.Expr = self._make_dynamic(T_static)
        self.V: sp.Expr = self._make_dynamic(V_static)
        self.L: sp.Expr = self._make_dynamic(L_static)
        
        
        self.eom: Optional[List[sp.Eq]] = None
        
        # Calculate equations of motion once during initialization
        self.equations_of_motion()
    
    def _parse_coordinates(self, T: str, V: str, coordinates: Union[str, List[str], None]) -> List[str]:
        """Parse and validate coordinate specifications."""
        default_coordinates = ['theta', 'alpha', 'x', 'y', 'z', 'p', 'q']
        
        if coordinates is None:
            # Auto-detect coordinates
            coords = [coord for coord in default_coordinates if coord in T or coord in V]
            print(f"Auto-detected coordinates: {', '.join(coords)}")
            return coords
        elif isinstance(coordinates, str):
            return [c.strip() for c in coordinates.replace(',', ' ').split()]
        elif isinstance(coordinates, list):
            return coordinates
        else:
            raise ValueError("Coordinates must be a string, list of strings, or None.")
    
    def _make_dynamic(self, expr: sp.Expr) -> sp.Expr:
        """Convert a static expression to a time-dependent one."""
        # Create coordinate functions and substitutions
        subs_dict: Dict[sp.Symbol, sp.Expr] = {}
        
        # Get all symbols in the expression
        symbols: Set[sp.Symbol] = expr.free_symbols
        symbol_names = {str(sym) for sym in symbols}
        
        # For each coordinate, create time functions and velocity variables
        for coord in self.coordinates:
            # Create functions q(t) and derivatives
            q_func = sp.Function(coord)(self.t)
            q_dot = q_func.diff(self.t)
            
            # Create substitution pairs
            if coord in symbol_names:
                subs_dict[sp.Symbol(coord)] = q_func
            
            # Handle velocity terms (x_dot or v_x notation)
            vel_symbols = [f"{coord}_dot", f"v_{coord}"]
            for vel_sym in vel_symbols:
                if vel_sym in symbol_names:
                    subs_dict[sp.Symbol(vel_sym)] = q_dot
        
        # Apply all substitutions
        return expr.subs(subs_dict)
    
    def equations_of_motion(self) -> List[sp.Eq]:
        """
        Derive and return the Euler-Lagrange equations for each coordinate.
        Caches the result for future calls.
        """
        if self.eom is None:
            self.eom = []
            
            for coord in self.coordinates:
                # Create coordinate function and its derivative
                q = sp.Function(coord)(self.t)
                q_dot = q.diff(self.t)
                
                # Compute partial derivatives for Lagrange's equations
                dL_dqdot = sp.diff(self.L, q_dot)
                d_dt_dL_dqdot = sp.diff(dL_dqdot, self.t)
                dL_dq = sp.diff(self.L, q)
                
                # Create the Euler-Lagrange equation
                euler_eq = sp.Eq(d_dt_dL_dqdot - dL_dq, 0)
                self.eom.append(euler_eq)
                
        return self.eom

    def print_equations_of_motion(self) -> None:
        """Print the equations of motion in a readable format."""
        eom = self.equations_of_motion()
        print("Equations of Motion:")
        for i, eq in enumerate(eom, start=1):
            print(f"Equation {i}:")
            print(sp.pretty(eq))
            print()