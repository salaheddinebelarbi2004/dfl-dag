"""
Polynomial Proxy Model
Pi(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
Encodes properties of gradient updates with reduced dimensionality
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import hashlib
from federated_learning_config import POLY_DEGREE, POLY_EVAL_POINTS


class PolynomialProxy:
    """
    Polynomial proxy that approximates gradient updates.
    Used to reduce ZKP cost through algebraic verification.
    
    Pi(x) encodes:
    - a0: baseline update magnitude
    - a1: primary gradient direction
    - a2+: higher-order properties (variance, curvature, etc.)
    """

    def __init__(self, degree: int = POLY_DEGREE):
        """
        Initialize polynomial proxy.
        
        Args:
            degree: Degree of the polynomial
        """
        self.degree = degree
        self.coefficients: np.ndarray = None  # a0, a1, ..., an
        self.eval_points: List[float] = None
        self.eval_values: List[float] = None

    def fit_from_gradient(self, gradient: np.ndarray,
                         model_param: np.ndarray = None) -> None:
        """
        Fit polynomial proxy from gradient updates.
        Uses curve fitting over carefully chosen evaluation points.

        Args:
            gradient: The gradient update vector ∆Wi
            model_param: Current model parameters Wt (optional)
        """
        # Normalize gradient if too large
        gradient_flat = gradient.flatten()
        grad_norm = float(np.linalg.norm(gradient_flat))
        if grad_norm > 0:
            gradient_flat = gradient_flat / grad_norm

        # Select evaluation points (0 to 1, stratified)
        self.eval_points = np.linspace(0, 1, POLY_EVAL_POINTS).tolist()

        # Compute values at evaluation points
        # Simulate: Pi(x) approximates magnitude distribution
        self.eval_values = []
        for x in self.eval_points:
            # Simulate polynomial evaluation: encode gradient statistics
            # Real implementation would use more sophisticated fitting
            if x == 0:
                value = grad_norm
            else:
                # Encode variance/higher moments
                value = grad_norm * np.exp(-x * np.sum(gradient_flat ** 2))
            self.eval_values.append(float(value))

        # Fit polynomial through evaluation points
        self.coefficients = np.polyfit(self.eval_points, self.eval_values,
                                       min(self.degree, len(self.eval_points) - 1))

    def evaluate(self, x: float) -> float:
        """
        Evaluate polynomial at point x: Pi(x)

        Args:
            x: Evaluation point

        Returns:
            Polynomial value at x
        """
        if self.coefficients is None:
            raise ValueError("Polynomial not fitted yet")
        return float(np.polyval(self.coefficients, x))

    def get_coefficients(self) -> np.ndarray:
        """Get polynomial coefficients [an, ..., a1, a0]"""
        return self.coefficients.copy()

    def get_coefficients_ordered(self) -> List[float]:
        """Get coefficients in order [a0, a1, ..., an]"""
        if self.coefficients is None:
            return []
        return list(reversed(self.coefficients))

    def verify_consistency(self, gradient: np.ndarray,
                          tolerance: float = 0.1) -> bool:
        """
        Verify that polynomial is consistent with gradient.
        Used in ZKP verification.

        Args:
            gradient: Original gradient update
            tolerance: Maximum allowed error

        Returns:
            True if polynomial is consistent
        """
        if self.coefficients is None:
            return False

        # Check if polynomial evaluation at sample points
        # is close to normalized gradient magnitude
        gradient_flat = gradient.flatten()
        grad_norm = float(np.linalg.norm(gradient_flat))

        if float(grad_norm) == 0:
            return np.allclose(self.evaluate(0), 0, atol=tolerance)

        # Test at random points
        test_points = np.linspace(0.1, 0.9, 5)
        errors = []
        for x in test_points:
            poly_val = self.evaluate(x)
            expected_val = grad_norm * np.exp(-x * np.sum((gradient_flat / grad_norm) ** 2))
            error = abs(poly_val - expected_val)
            errors.append(error)

        mean_error = float(np.mean(errors))
        return float(mean_error) <= float(tolerance * grad_norm)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize polynomial to dictionary"""
        return {
            "degree": self.degree,
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else None,
            "eval_points": self.eval_points,
            "eval_values": self.eval_values,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PolynomialProxy':
        """Deserialize polynomial from dictionary"""
        poly = PolynomialProxy(degree=data["degree"])
        if data["coefficients"]:
            poly.coefficients = np.array(data["coefficients"])
        poly.eval_points = data["eval_points"]
        poly.eval_values = data["eval_values"]
        return poly

    def get_hash(self) -> str:
        """Get hash of polynomial coefficients for commitment verification"""
        if self.coefficients is None:
            return hashlib.sha256(b"").hexdigest()
        coeff_bytes = self.coefficients.tobytes()
        return hashlib.sha256(coeff_bytes).hexdigest()


class PolynomialProxyFactory:
    """Factory for creating and managing polynomial proxies"""

    @staticmethod
    def create_from_update(gradient: np.ndarray,
                          model_params: np.ndarray = None,
                          degree: int = POLY_DEGREE) -> PolynomialProxy:
        """Create polynomial proxy from gradient update"""
        proxy = PolynomialProxy(degree=degree)
        proxy.fit_from_gradient(gradient, model_params)
        return proxy

    @staticmethod
    def batch_create(gradients: List[np.ndarray],
                    degree: int = POLY_DEGREE) -> List[PolynomialProxy]:
        """Create multiple polynomial proxies in batch"""
        proxies = []
        for grad in gradients:
            proxy = PolynomialProxy(degree=degree)
            proxy.fit_from_gradient(grad)
            proxies.append(proxy)
        return proxies
