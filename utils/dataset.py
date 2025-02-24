from typing import Tuple
from torch.utils.data import Dataset
from torch import Tensor


class WeatherDataset(Dataset):
    def __init__(self, surface: Tensor, upper: Tensor, step: int = 1):
        self.step = step
        self.surface = surface
        self.upper = upper

    def __len__(self) -> int:
        # return len(self.surface) - self.step
        return self.upper.size(0) - self.step

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        return (self.surface[idx], self.upper[idx]), (
            self.surface[idx + self.step],
            self.upper[idx + self.step],
        )
