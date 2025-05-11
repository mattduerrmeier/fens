import torch
from autoencoder.model import Autoencoder, Decoder


def train_student(
    student_model: Decoder,
    teacher_model: Autoencoder,
    epochs: int,
    epoch_size: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
):
    batches = epoch_size // batch_size

    mse_loss = torch.nn.MSELoss()

    def adapted_loss(
        actual_output: tuple[torch.Tensor, torch.Tensor],
        expected_output: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return mse_loss(actual_output[0], expected_output[0]) + 2 * mse_loss(
            actual_output[1], expected_output[1].float()
        )

    loss_function = adapted_loss

    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(batches):
            latent = teacher_model.sample_latent(epoch_size).detach()
            optimizer.zero_grad()

            teacher_output = teacher_model.sample_from_latent(latent)
            student_output = student_model.sample_from_latent(latent)

            loss = loss_function(student_output, teacher_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch} Loss: {total_loss / batches :.3f}")
