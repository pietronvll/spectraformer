import jax
import optax
from flax.training.train_state import TrainState


@jax.jit
def train_step(state: TrainState, batch, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,  # Disable dropout for the moment
            rngs={"dropout": dropout_train_key},
        )
        loss = optax.squared_error(pred_spectra, batch["spectra"]).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
