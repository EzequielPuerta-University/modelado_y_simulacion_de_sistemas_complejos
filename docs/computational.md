# Modelos Computacionales

* [Schelling](#schelling)
* [Mercado Inmobiliario](#mercado-inmobiliario)
* [Condensación](#condensación)
* [Juego de la Vida (Conway)](#juego-de-la-vida-conway)

---

## Schelling

```python
from src.models.computational.schelling.model import Schelling
```

Parámetros:
* `tolerance: int`

## Mercado Inmobiliario

```python
from src.models.computational.real_state_market.model import RealStateMarket
```

Parámetros:
* `alpha: float = 0.5`
* `A: float = 1 / 16`
* `B: float = 0.5`
* `utility_tolerance: float = 0.85`

## Condensación

```python
from src.models.computational.condensation.model import Condensation
```

Parámetros:
* `probability: float`

## Juego de la vida (Conway)

```python
from src.models.computational.game_of_life.model import GameOfLife
from src.models.computational.game_of_life.seeds import Seed
```

Parámetros:
* `seeds: List[Seed]`

> [Volver](../README.md)
