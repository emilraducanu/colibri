def distilbert(
    epochs: int,
    batch_size_: int,
    learning_rate: float,
    dropout: float,
    padding_length: int,
    testset_size: float,
):
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
    from sklearn.metrics import f1_score

    # Load pre-trained DistilBERT model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    # Convert columns to list
    df = pd.read_pickle("/home/er/Documents/Cirad/colibri/data/trainset/trainset.pkl")
    titles = df["Title"].tolist()
    abstracts = df["Abstract"].tolist()
    keywords = df["Keywords"].tolist()

    # Tokenize and encode the titles, abstracts, and keywords
    titles = tokenizer(
        titles,
        padding="max_length",
        truncation=True,
        max_length=padding_length,
        return_tensors="pt",
    )
    abstracts = tokenizer(
        abstracts,
        padding="max_length",
        truncation=True,
        max_length=padding_length,
        return_tensors="pt",
    )
    keywords = tokenizer(
        keywords,
        padding="max_length",
        truncation=True,
        max_length=padding_length,
        return_tensors="pt",
    )

    # Create attention masks for each input
    titles_attention_masks = titles["attention_mask"]
    abstracts_attention_masks = abstracts["attention_mask"]
    keywords_attention_masks = keywords["attention_mask"]

    # Prepare labels (assuming 'Screening' column has 'included' and 'excluded' values)
    labels = torch.tensor(df["Screening"].map({"included": 1, "excluded": 0}).values)

    # Split data into training and validation sets
    (
        train_titles,
        val_titles,
        train_titles_attention_mask,
        val_titles_attention_mask,
        train_labels,
        val_labels,
    ) = train_test_split(
        titles["input_ids"],
        titles_attention_masks,
        labels,
        test_size=testset_size,
        random_state=42,
    )

    (
        train_abstracts,
        val_abstracts,
        train_abstracts_attention_mask,
        val_abstracts_attention_mask,
    ) = train_test_split(
        abstracts["input_ids"],
        abstracts_attention_masks,
        test_size=testset_size,
        random_state=42,
    )

    (
        train_keywords,
        val_keywords,
        train_keywords_attention_mask,
        val_keywords_attention_mask,
    ) = train_test_split(
        keywords["input_ids"],
        keywords_attention_masks,
        test_size=testset_size,
        random_state=42,
    )

    # Create DataLoader for training and validation sets for each input type
    train_titles_dataset = TensorDataset(
        train_titles, train_titles_attention_mask, train_labels
    )
    val_titles_dataset = TensorDataset(
        val_titles, val_titles_attention_mask, val_labels
    )

    train_abstracts_dataset = TensorDataset(
        train_abstracts, train_abstracts_attention_mask, train_labels
    )
    val_abstracts_dataset = TensorDataset(
        val_abstracts, val_abstracts_attention_mask, val_labels
    )

    train_keywords_dataset = TensorDataset(
        train_keywords, train_keywords_attention_mask, train_labels
    )
    val_keywords_dataset = TensorDataset(
        val_keywords, val_keywords_attention_mask, val_labels
    )

    train_titles_dataloader = DataLoader(
        train_titles_dataset, batch_size=batch_size_, shuffle=True
    )
    val_titles_dataloader = DataLoader(val_titles_dataset, batch_size=batch_size_)

    train_abstracts_dataloader = DataLoader(
        train_abstracts_dataset, batch_size=batch_size_, shuffle=True
    )
    val_abstracts_dataloader = DataLoader(val_abstracts_dataset, batch_size=batch_size_)

    train_keywords_dataloader = DataLoader(
        train_keywords_dataset, batch_size=batch_size_, shuffle=True
    )
    val_keywords_dataloader = DataLoader(val_keywords_dataset, batch_size=batch_size_)

    # Define the classification heads
    titles_classification_head = nn.Sequential(
        nn.Linear(model.config.hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 2),
    )

    abstracts_classification_head = nn.Sequential(
        nn.Linear(model.config.hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 2),
    )

    keywords_classification_head = nn.Sequential(
        nn.Linear(model.config.hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 2),
    )

    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizers for each classification head
    optimizer_titles = torch.optim.AdamW(
        titles_classification_head.parameters(), lr=learning_rate
    )
    optimizer_abstracts = torch.optim.AdamW(
        abstracts_classification_head.parameters(), lr=learning_rate
    )
    optimizer_keywords = torch.optim.AdamW(
        keywords_classification_head.parameters(), lr=learning_rate
    )

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Create a tqdm progress bar for the training loop
        train_progress_bar = tqdm(
            zip(
                train_titles_dataloader,
                train_abstracts_dataloader,
                train_keywords_dataloader,
                train_labels,
            ),
            total=len(train_titles_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs} (Training)",
            leave=False,
        )

        for (
            batch_titles,
            batch_abstracts,
            batch_keywords,
            batch_labels,
        ) in train_progress_bar:
            (
                batch_titles,
                batch_titles_attention_mask,
                batch_labels_titles,
            ) = batch_titles
            (
                batch_abstracts,
                batch_abstracts_attention_mask,
                batch_labels_abstracts,
            ) = batch_abstracts
            (
                batch_keywords,
                batch_keywords_attention_mask,
                batch_labels_keywords,
            ) = batch_keywords

            optimizer_titles.zero_grad()
            optimizer_abstracts.zero_grad()
            optimizer_keywords.zero_grad()

            with torch.no_grad():
                titles_outputs = model(
                    input_ids=batch_titles, attention_mask=batch_titles_attention_mask
                )
                abstracts_outputs = model(
                    input_ids=batch_abstracts,
                    attention_mask=batch_abstracts_attention_mask,
                )
                keywords_outputs = model(
                    input_ids=batch_keywords,
                    attention_mask=batch_keywords_attention_mask,
                )

            titles_logits = titles_classification_head(
                titles_outputs.last_hidden_state[:, 0, :]
            )
            abstracts_logits = abstracts_classification_head(
                abstracts_outputs.last_hidden_state[:, 0, :]
            )
            keywords_logits = keywords_classification_head(
                keywords_outputs.last_hidden_state[:, 0, :]
            )

            combined_logits = titles_logits + abstracts_logits + keywords_logits

            loss_titles = loss_fn(titles_logits, batch_labels_titles)
            loss_abstracts = loss_fn(abstracts_logits, batch_labels_abstracts)
            loss_keywords = loss_fn(keywords_logits, batch_labels_keywords)
            loss = loss_titles + loss_abstracts + loss_keywords
            loss.backward()

            optimizer_titles.step()
            optimizer_abstracts.step()
            optimizer_keywords.step()

            train_loss += loss.item()

        # Update the tqdm progress bar description with average loss
        train_progress_bar.set_postfix(
            train_loss=f"{train_loss / len(train_titles_dataloader):.4f}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        f1_scores = []

        # Create a tqdm progress bar for the validation loop
        val_progress_bar = tqdm(
            zip(
                val_titles_dataloader,
                val_abstracts_dataloader,
                val_keywords_dataloader,
                val_labels,
            ),
            total=len(val_titles_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)",
            leave=False,
        )

        # Check if GPU is available and set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for (
                batch_titles,
                batch_abstracts,
                batch_keywords,
                batch_labels,
            ) in val_progress_bar:
                # Unpack the data for each input type
                (
                    titles_input_ids,
                    titles_attention_mask,
                    batch_labels_titles,
                ) = batch_titles
                (
                    abstracts_input_ids,
                    abstracts_attention_mask,
                    batch_labels_abstracts,
                ) = batch_abstracts
                (
                    keywords_input_ids,
                    keywords_attention_mask,
                    batch_labels_keywords,
                ) = batch_keywords

                # Send labels to the device
                batch_labels_titles = batch_labels_titles.to(device)
                batch_labels_abstracts = batch_labels_abstracts.to(device)
                batch_labels_keywords = batch_labels_keywords.to(device)

                titles_outputs = model(
                    input_ids=titles_input_ids.to(device),
                    attention_mask=titles_attention_mask.to(device),
                )
                abstracts_outputs = model(
                    input_ids=abstracts_input_ids.to(device),
                    attention_mask=abstracts_attention_mask.to(device),
                )
                keywords_outputs = model(
                    input_ids=keywords_input_ids.to(device),
                    attention_mask=keywords_attention_mask.to(device),
                )

                titles_logits = titles_classification_head(
                    titles_outputs.last_hidden_state[:, 0, :]
                )
                abstracts_logits = abstracts_classification_head(
                    abstracts_outputs.last_hidden_state[:, 0, :]
                )
                keywords_logits = keywords_classification_head(
                    keywords_outputs.last_hidden_state[:, 0, :]
                )

                # Calculate the overall logits by combining the logits from different input types
                combined_logits = (
                    titles_logits + abstracts_logits + keywords_logits
                ) / 3

                # Convert predictions to NumPy array
                _, predicted = torch.max(combined_logits, 1)
                predicted = predicted.cpu().numpy()

                # Calculate F1-score for each batch
                f1 = f1_score(
                    batch_labels_titles.cpu(),
                    predicted,
                    average="macro",
                    zero_division=1,
                )
                f1_scores.append(f1)

                # Calculate the loss for each input type
                loss_titles = loss_fn(titles_logits, batch_labels_titles)
                loss_abstracts = loss_fn(abstracts_logits, batch_labels_abstracts)
                loss_keywords = loss_fn(keywords_logits, batch_labels_keywords)

                # Calculate the total loss as a combination of losses from different input types
                loss = loss_titles + loss_abstracts + loss_keywords
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_titles_dataloader)
        avg_f1 = np.mean(f1_scores)

        # Update the tqdm progress bar description with validation results
        val_progress_bar.set_postfix(
            val_loss=f"{avg_val_loss:.4f}", f1_score=f"{avg_f1:.4f}"
        )

        # Print the F1-score at the end of each epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation F1-score: {avg_f1:.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_model.pt")

    return avg_f1
