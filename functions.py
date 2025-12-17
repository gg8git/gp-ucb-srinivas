import torch
from abc import ABC, abstractmethod
import math

class SyntheticFunction(ABC):
    """Abstract base class for synthetic black-box functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the function."""
        pass

    @property
    @abstractmethod
    def bounds(self) -> torch.Tensor:
        """Bounds of the input domain as a tensor of shape (2, D)."""
        pass

    @property
    @abstractmethod
    def star(self) -> float:
        """Known global minimum value (optional for regret computation)."""
        pass

    @abstractmethod
    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the function at points x."""
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class Branin(SyntheticFunction):
    @property
    def name(self):
        return "Branin"

    @property
    def bounds(self):
        return torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=torch.float32)

    @property
    def star(self):
        return 0.397887

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]
        a = 1.0
        b = 5.1 / (4 * math.pi ** 2)
        c = 5 / math.pi
        r = 6.0
        s = 10.0
        t = 1 / (8 * math.pi)
        return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s


class Ackley(SyntheticFunction):
    @property
    def name(self):
        return "Ackley"

    @property
    def bounds(self):
        return torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=torch.float32)

    @property
    def star(self):
        return 0.0

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        a = 20
        b = 0.2
        c = 2 * math.pi
        term1 = -a * torch.exp(-b * torch.sqrt(0.5 * (x1**2 + x2**2)))
        term2 = -torch.exp(0.5 * (torch.cos(c*x1) + torch.cos(c*x2)))
        return term1 + term2 + a + math.e


class Levy(SyntheticFunction):
    @property
    def name(self):
        return "Levy"

    @property
    def bounds(self):
        return torch.tensor([[-10.0, -10.0], [10.0, 10.0]], dtype=torch.float32)

    @property
    def star(self):
        return 0.0

    def f(self, x: torch.Tensor) -> torch.Tensor:
        w1 = 1 + (x[..., 0] - 1)/4
        w2 = 1 + (x[..., 1] - 1)/4
        term1 = torch.sin(math.pi*w1)**2
        term2 = (w1 - 1)**2 * (1 + 10*torch.sin(math.pi*w1 + 1)**2)
        term3 = (w2 - 1)**2 * (1 + torch.sin(2*math.pi*w2)**2)
        return term1 + term2 + term3


class Himmelblau(SyntheticFunction):
    @property
    def name(self):
        return "Himmelblau"

    @property
    def bounds(self):
        return torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=torch.float32)

    @property
    def star(self):
        return 0.0

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2