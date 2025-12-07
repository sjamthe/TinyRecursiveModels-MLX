import copy
import mlx.nn as nn
import mlx.core as mx


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def _iter_params(self, mapping, prefix=""):
        """Yield (full_name, leaf_param) for nested dict-like parameter containers."""
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from self._iter_params(v, new_prefix)
                else:
                    yield new_prefix, v
        else:
            yield prefix, mapping

    def _set_param_by_name(self, mapping, name, value):
        """Set a leaf parameter value given its dotted name inside a nested mapping."""
        parts = name.split(".")
        cur = mapping
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value

    def _is_trainable(self, p):
        # MLX arrays may not have requires_grad; assume trainable if attribute missing
        try:
            return bool(getattr(p, "requires_grad", True))
        except Exception:
            return True

    def _to_mx_array(self, p):
        """Try to coerce p to an mx array; return None if not possible."""
        try:
            return mx.array(p)
        except Exception:
            return None

    def register(self, module):
        params_map = module.parameters()
        for name, param in self._iter_params(params_map):
            # If the "leaf" itself is a dict (defensive), iterate into it
            if isinstance(param, dict):
                for sub_name, sub_param in self._iter_params(param, prefix=name):
                    if not self._is_trainable(sub_param):
                        continue
                    arr = self._to_mx_array(sub_param)
                    if arr is not None:
                        self.shadow[sub_name] = arr
                continue

            if not self._is_trainable(param):
                continue
            arr = self._to_mx_array(param)
            if arr is not None:
                # store a copy as the shadow
                self.shadow[name] = arr

    def update(self, module):
        params_map = module.parameters()
        for name, param in self._iter_params(params_map):
            if isinstance(param, dict):
                for sub_name, sub_param in self._iter_params(param, prefix=name):
                    if not self._is_trainable(sub_param):
                        continue
                    p_arr = self._to_mx_array(sub_param)
                    if p_arr is None:
                        continue
                    shadow = self.shadow.get(sub_name)
                    if shadow is None:
                        self.shadow[sub_name] = mx.array(p_arr)
                    else:
                        self.shadow[sub_name] = (1.0 - self.mu) * p_arr + self.mu * shadow
                continue

            if not self._is_trainable(param):
                continue
            p_arr = self._to_mx_array(param)
            if p_arr is None:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                self.shadow[name] = mx.array(p_arr)
            else:
                self.shadow[name] = (1.0 - self.mu) * p_arr + self.mu * shadow

    def ema(self, module):
        """Overwrite module parameters with EMA (in-place in the nested mapping)."""
        params_map = module.parameters()
        for name, _ in self._iter_params(params_map):
            if name in self.shadow:
                # set a copy of shadow back into module parameters
                self._set_param_by_name(params_map, name, mx.array(self.shadow[name]))

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
