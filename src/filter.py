def train_distilbert(config):
    """Quick description

    Long description

    Parameters:
    query (str):

    Returns:

    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from transformers import DistilBertTokenizer, DistilBertModel
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    from sklearn.metrics import f1_score, accuracy_score
    import pytz
    import datetime
    import time
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    start_time = time.time()

    # Create timestamp
    utc_current_time = datetime.datetime.now(tz=pytz.timezone("UTC"))
    utc_current_time_str = utc_current_time_str = utc_current_time.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # Create output dir
    current_dir = os.getcwd()
    target_folder = "colibri"
    while os.path.basename(current_dir) != target_folder:
        current_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(current_dir, "data/distilbert_runs/" + utc_current_time_str)
    os.makedirs(data_dir)

    # Define device (GPU is available, CPU otherwise)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    # Custom tokenizer vocabulary
    new_tokens = [
        "SOC",
        "SOM",
        "meta-analysis",
        "metaanalysis",
        "meta-analyses",
        "metaanalyses",
    ]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Import training set
    df = pd.read_pickle(config["distilbert_trainset_path"])
    df["Text"] = df["Title"] + " " + df["Abstract"] + " " + df["Keywords"]
    df["Text"] = df["Text"].astype(str)
    text_list = df["Text"].tolist()
    df = df[["Screening", "Text"]]

    # Tokenize and encode the text
    text = tokenizer(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=config["padding_length"],
        return_tensors="pt",
    )

    # Create attention masks
    attention_masks = text["attention_mask"]

    # Prepare labels
    labels = torch.tensor(df["Screening"].map({"included": 1, "excluded": 0}).values)

    # Split data into training and validation sets
    (
        train_text,
        val_text,
        train_attention_masks,
        val_attention_masks,
        train_labels,
        val_labels,
    ) = train_test_split(
        text["input_ids"],
        attention_masks,
        labels,
        test_size=config["testset_size"],
        random_state=42,
        shuffle=False,
    )

    # Create DataLoader for training and validation sets for each input type
    train_dataset = TensorDataset(train_text, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_text, val_attention_masks, val_labels)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Define the classification head
    classification_head = nn.Sequential(
        nn.Linear(model.config.hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(64, 2),
    )

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.AdamW(
        classification_head.parameters(), lr=config["learning_rate"]
    )

    # TRAINING LOOP
    num_epochs = config["epochs"]
    f1_scores_over_epochs = []
    accuracy_score_over_epochs = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Create progress bar for the training loop
        train_progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs} (Training)",
            leave=False,
        )
        for i in train_progress_bar:
            # Unpack data
            (batch_text, batch_attention_masks, batch_labels) = i
            optimizer.zero_grad()

            # Training of the model
            with torch.no_grad():
                text_outputs = model(
                    input_ids=batch_text, attention_mask=batch_attention_masks
                )

            text_logits = classification_head(text_outputs.last_hidden_state[:, 0, :])
            loss = loss_fn(text_logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Update progress bar
        train_progress_bar.set_postfix(
            train_loss=f"{train_loss / len(train_dataloader):.4f}"
        )

        # VALIDATION LOOP
        model.eval()
        predictions = [[], [], []]

        # Create progress bar for the validation loop
        val_progress_bar = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)",
            leave=False,
        )

        for i in val_progress_bar:
            # Unpack data
            (batch_text, batch_attention_masks, batch_labels) = i

            # Send labels to the device
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                # Validation of the model
                outputs = model(
                    input_ids=batch_text.to(device),
                    attention_mask=batch_attention_masks.to(device),
                )

            logits = classification_head(outputs.last_hidden_state[:, 0, :])
            _, predicted = torch.max(logits, 1)
            for index, tsr in enumerate(batch_labels):
                predictions[0].append(tsr.item())
            for index, tsr in enumerate(predicted):
                predictions[1].append(tsr.item())

        # Calculate F1-score
        current_epoch_f1_score = f1_score(
            predictions[0], predictions[1], zero_division=1, average="weighted"
        )
        current_epoch_accuracy = accuracy_score(predictions[0], predictions[1])
        f1_scores_over_epochs.append(current_epoch_f1_score)
        accuracy_score_over_epochs.append(current_epoch_accuracy)

        # Update progress bar description with validation results
        val_progress_bar.set_postfix(f1_score=f"{current_epoch_f1_score:.4f}")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Validation | Accuracy: {current_epoch_accuracy:.4f}, F1-score: {current_epoch_f1_score:.4f}"
        )

    # Save the fine-tuned model
    saved_model_path = os.path.join(data_dir, "fine_tuned_model.pt")
    torch.save(model.state_dict(), saved_model_path)

    # # Save predictions over validation set
    # predictions_df = pd.DataFrame(predictions)
    # predictions_csv_path = os.path.join(data_dir, "predictions.csv")
    # predictions_df.to_csv(predictions_csv_path, index=False)

    # Save confusion matrix
    cm = confusion_matrix(
        predictions[0],
        predictions[1],
        labels=[0, 1],
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion matrix")
    plt.savefig(os.path.join(data_dir, "confusion_matrix.png"))
    plt.close()

    # Save graph of F1 score evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config["epochs"] + 1), f1_scores_over_epochs, marker="o")
    plt.title("F1-score evolution")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid()
    plt.savefig(os.path.join(data_dir, "f1-score_evolution.png"))
    plt.close()

    # Save run info
    end_time = time.time()
    duration = end_time - start_time
    config["last_f1_score"] = f1_scores_over_epochs[-1]
    config["run_duration"] = duration
    with open(os.path.join(data_dir, "run_info.json"), "w") as config_file:
        json.dump(config, config_file)

    return 1