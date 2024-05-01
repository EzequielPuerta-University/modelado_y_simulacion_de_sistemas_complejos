# Modelo abstracto

* [Grilla abstracta](#grilla-abstracta)
* [Agente abstracto](#agente-abstracto)
* [Evolución del modelo](#evolución-del-modelo)
* [Series](#series)

---

## Grilla abstracta

```python
from src.models.abstract.model import AbstractLatticeModel
```

Incluye todo el comportamiento básico de un modelo computable del tipo autómata celular sobre de 2 dimensiones.

Espera parámetros clásicos como:
* `length`: un entero $n$ que define el tamaño final de la grilla $n \times n$.
* `configuration`: una configuración inicial de la grilla. De no proveer ninguna, se genera una automáticamente.
* `neighborhood`: la estrategia que define dada una posición, cuales celdas serán sus vecinas. Hay implementadas dos posibles:
    * `src.simulation.core.neighborhood.VonNeumann` (default)
    * `src.simulation.core.neighborhood.Moore`
* `agent_types`: la cantidad de tipos de agentes disponibles en la grilla. Valor por default: 2.
* `update_simultaneously`: valor *booleano* que permite indicar al modelo si la actualización de cada agente debe impactar en la configuración global, permitiendo que impacte en las actualizaciones siguientes (`False`, valor por defecto), o si en un mismo paso de simulación se actualizan todos los agentes en una grilla temporal y al finalizar dicho proceso, se impactan todos los cambios temporales en la grilla global, logrando así que cada agente se base en el estado previo (`True`).

## Agente abstracto

```python
from src.models.abstract.agent import Agent
```

Si bien al modelo abstracto se le pueden ingresar configuraciones iniciales (un *array* de $x \in [0..$`agent_types`$)$ de $n \times n$), la grilla sobre la que trabaja el modelo no termina siendo una matriz de enteros, sino que es una de *agentes*, por lo que pueden tener un comportamiento mas complejo de ser necesario.

El modelo abstracto provee un agente abstracto básico, que contiene un único atributo `agent_type`. Por lo tanto, si en la configuración ingresada $M_{T_0}$, tenemos que $M_{T_0}[i,j] = x$, con $x \in [0..$`agent_types`$)$, en la configuración inicial $M_{T_1}$ realmente tendremos que $M_{T_1}[i,j] = $`Agent(agent_type=x)`$.

Luego, si el modelo $Y$ necesita un comportamiento más complejo de parte los agentes (ver el caso del modelo de mercado inmobiliario), se puede crear $W$, que subclasifica la clase `Agent` y especializarlo para el caso. Cuando se inicializa el modelo abstracto, el mismo hace uso del método `_create_agent`, por lo que se puede reimplementar éste método en $Y$ para hacer uso del nuevo modelo de agente:

```python
from src.models.abstract.agent import Agent
from src.models.abstract.model import AbstractLatticeModel

class W(Agent):
    pass

class Y(AbstractLatticeModel):
    ...
    def _create_agent(self, basic_agent: Agent, i: int, j: int) -> W:
        ...
        return W(agent_type=basic_agent.agent_type)
    ...
```

## Evolución del modelo

Como se mencionó, toda la lógica básica del autómata está contenida en el modelo abstracto. El único método que obligatoriamente debe definir un modelo concreto es el `step`. En el mismo se debe indicar como se actualizará el modelo, agente a agente. Luego, según el valor de `update_simultaneously`, esa actualización impactará en la configuración global o una temporal.

```python
def step(self, i: int, j: int, **kwargs) -> None:
    pass
```

## Series

La evolución del modelo irá generando múltiples datos de interés, iteración a iteración. Se pueden implementar diferentes métodos en el modelo concreto y decorarlos con `@as_series`. Esto hará que luego de cada iteración global, el resultado de dicho método se almacene, generando una serie de datos para poder analizar al finalizar la simulación. El resultado puede ser cualquier objeto, ya sea valores enteros, decimales o hasta snapshots de la grilla.

De hecho, una serie que puede ser de interés trivial, es la grilla y sus valores de `agent_type` para cada celda, en un tiempo dado. El siguiente método `agent_types_lattice`, nos asegurará tener esa información al finalizar la simulación:

```python
from src.models.abstract.model import AbstractLatticeModel, as_series

class Y(AbstractLatticeModel):
    ...
    @as_series
    def agent_types_lattice(self) -> List[List[int]]:
        action = lambda i, j: self.get_agent(i, j).agent_type
        return self._process_lattice_with(action)
```

La serie se guardará en un atributo `series: dict`, de la instancia del modelo simulado. La *key* de la serie es el nombre del método relacionado con la misma. Es decir si la configuración inicial es $M_{T_0}$, para la serie del caso anterior vale:

`Y().series["agent_types_lattice"][0] = ` $M_{T_0}$

En algunos casos puede ser útil relacionar los datos obtenidos en una serie con cierta *metadata* a visualizar en los *plotters*. Esto se puede lograr usando el decorador `as_series_with`.

> [!IMPORTANT]
> Ésto puede tener un costo computacional importante, en espacio y tiempo. No es la intención del presente código lograr optimizar este procedimiento, por lo tanto se debe utilizar con criterio.

> [Volver](../README.md)
