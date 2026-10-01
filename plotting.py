from pathlib import Path
import matplotlib.pyplot as plt


def plot_losses(train_losses: list,
                valid_losses: list,
                name: str,
                save_path: str) -> None:
    """
    Plots the training and validation losses over epochs and saves the plot to a specified path.

    Parameters:
        train_losses: list
            A list of training losses for each epoch.
        valid_losses: list
            A list of validation losses for each epoch.
        name: str
            The name of the model or experiment, used for the plot title.
        save_path: str
            The file path where the plot will be saved.
    """
    # The first element in train_losses and valid_losses is the pre-training loss.
    epochs = range(len(train_losses))

    plt.figure(figsize=(7, 5))

    plt.plot(epochs, train_losses, marker = "o", label = "Training")
    plt.plot(epochs, valid_losses, marker = "o", label = "Validation")
    
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss per token")
    plt.title(f"{name} loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha = 0.3)

    # Create parent directory if necessary.
    Path(save_path).parent.mkdir(parents = True, exist_ok = True)
    # bbox_inches = "tight" minimizes whitespace around the figure.
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    
    plt.close()


def plot_loss_comparison(losses: dict, save_path: str) -> None:
    """
    Plots the training and validation losses for multiple models for comparison.

    Parameters:
        losses: dict
            A dictionary where keys are model names and values are tuples of (train_losses, valid_losses).
        save_path: str
            The file path where the plot will be saved.
    """
    plt.figure(figsize = (9, 6))

    for name, (train_losses, valid_losses) in losses.items():
        epochs = range(len(train_losses))

        train_line, = plt.plot(epochs, train_losses, marker = "o", label = f"{name} - Training")
        # For the validation loss curve, reuse the same color but make the line dashed.
        color = train_line.get_color()
        plt.plot(epochs, valid_losses, marker = "o", linestyle = "--", color = color, label = f"{name} - Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss per token")
    plt.title("Training and validation loss")
    plt.xticks(epochs) # Assumes every model has the same number of epochs.
    plt.legend()
    plt.grid(alpha = 0.3)

    # Create parent directory if necessary.
    Path(save_path).parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")

    plt.close()