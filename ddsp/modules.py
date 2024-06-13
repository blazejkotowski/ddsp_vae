class SynthParameters(object):
  """
  A parameter of a synthesis model
  """

  def __init__(self, name: str, length: int):
    self._name = name
    self._length = length


class ParametersRegister(object):
  """
  Keeps track of the parameters of a synthesis model
  """
  def __init__(self):
    self._parameters = {}


  def register(self, name: str, length: int):
    """
    Register a parameter
    Args:
      - name: str, the name of the parameter
      - shape: Tuple, the shape of the parameter
      - dtype: str, the data type of the parameter
    """
    self._parameters[name] = SynthParameters(name, length)


  def n_parameters(self) -> int:
    """
    Get the number of parameters
    Returns:
      - int, the number of parameters
    """
    sum([p._length for p in self._parameters.values()])
