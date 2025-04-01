import jax
import jax.numpy as jnp

@jax.pmap
def dummy_pmap(x):
    jax.debug.print("DUMMY PMAP WORKING - input shape: {}", x.shape)
    return x + 1

# Try with known-good input
dummy_input = jnp.ones((jax.device_count(), 1))  # (4,1) for 4 devices
print("Dummy input:", dummy_input)
print("Dummy pmap result:", dummy_pmap(dummy_input))