## `plot_docs`: **Documentation for `Plot`** 
##### [link to `plot.py`](https://github.com/DolevArtzi/algsforbigdata/blob/main/plot.py)
### `benchmark_np`

This function is used to benchmark a numpy function on vectors, matrices, or tensors, and create a graph of its performance, optionally comparing to theoretical bounds as well. 

#### Arguments:
- `f : function`: The numpy function to benchmark. It takes either an array (`'a`) or a matrix/tensor (`'a`, usually `floats`) as input.
- `data : (list['a'] | None)`: Data to pass directly to `f` if not `None`.
- `type_ : str`: Specifies the type of input data to generate. Possible values are `'vec'`, `'sqmtx'`, `'mnmtx'`, or `'tensor'`.
- `gen_info :  (dict | None)`: Used to describe the inputs to generate for `f` if no data is provided directly.
    - possible arguments:
        - `type`: `'vec' | 'sqmtx' | 'mnmtx' | 'tensor'`
        - `range`: `list | tuple`, the range of sizes. See `op`/`delta` for how it's used in different cases
        - `base`: `number | tuple`, the base size to start at, for each dimension
        - `op`: `'mult' | 'add'`, the type of operation to incr. the size; if `'mult'`, then `next_size = curr_size * delta`, otherwise `next_size = curr_size + delta`
        - `delta`: `number`, defined as above
        - `dims`: `'N' | 'NN' | 'MN' | i, i >= 3`
            - the dimensions we expect, depending on `type`, only needs to be included if `type` is `'tensor'` and the desired dimension is larger than `3`
        -  `rand`: `str`: the type of randomness to use, defaults to `base` which uses `random.random,` in `[0,1)`, or the name of a probability distribution, e.g. `exponential` or `normal`
        - `randargs`: arguments to pass to the probability distribution, must be included if `rand != 'base'` and you don't want to use the default values for the distribution you chose, e.g. you want `'normal'` but not `Normal(0,1)`

#### **Default Values**:
- see the docs below for `_get_default_np_gen_info` for default values for `gen_info` depending on the type
## TODO: add override to benchmark_np.. and theoretical bounds graphing as well

#### Description:
- The function benchmarks a numpy function `f` on input data.
- It can generate input data based on the specified type and default parameters if no data is provided directly.
- Custom input generation information can be passed using the `gen_info` parameter.

### `_get_default_np_gen_info`

This function is used to generate default information for numpy array generation based on the given type and optional keyword overrides.

#### **Parameters**:
- `type_ : str`: Specifies the type of array to generate. Possible values are `'vec'`, `'tensor'`, `'mtx'`, or `'sqmtx'`.
- `**kw_override`: Optional keyword arguments to override default values.
    - if in `benchmark_np` you included a non-default usage of a random variable class, you must include a field `rand_params:[*args]` in `gen_info` so that `kw_override` will include it
    - if `type_ == 'mnmtx'`, include a field `mn:(m_base,n_base)` in `gen_info`

#### **Returns**:
- `gen_info : dict`: A dictionary containing the default generation information for numpy arrays.

#### Default Values:
```python
gen_info = {
    'range': [5, 13],
    'base': 2,
    'op': 'mult',
    'delta': 2,
    'rand': 'base'
}
