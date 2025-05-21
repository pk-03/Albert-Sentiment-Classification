
# Hindi Sentiment Classification using DeBERTa

This project fine-tunes the `albert-base-v2` transformer model for sentiment classification on Hindi text. The goal is to classify Hindi reviews into one of three sentiment classes: **positive**, **negative**, or **neutral**.

---

## 🧠 Model Architecture

- **Model**: [albert-base-v2](https://huggingface.co/albert/albert-base-v2)
- **Tokenizer**: AutoTokenizer from Hugging Face
- **Framework**: PyTorch with HuggingFace Transformers & Trainer API

---

## 📁 Dataset Format

The dataset should be a `.csv` or `.tsv` file with the following two columns:

- `Reviews`: Hindi review text.
- `labels`: Sentiment labels (e.g., `"positive"`, `"negative"`, `"neutral"`)

Example:

| Reviews                  | labels   |
|--------------------------|----------|
| यह फोन बहुत अच्छा है।   | positive |
| बैटरी खराब है।          | negative |
| यह एक औसत उत्पाद है।     | neutral  |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/pk-03/Albert-Sentiment-Classification.git
cd Albert-Sentiment-Classification
```

### 2. Install Dependencies
Make sure Python 3.8+ is installed.

```bash
pip install transformers torch pandas scikit-learn indic-nlp-library
```

---

## 🏋️‍♀️ Training the Model

### 1. Load and preprocess your dataset
Ensure your `DataFrame` has `Reviews` and `labels` columns or you can access the datasets given in [Datasets](https://github.com/pk-03/Data-Augmentation-and-Datasets.git).

```python
data = pd.read_csv("your_dataset.csv")  # or .tsv
```


### 2. Run the training script
The script:
- Encodes text with Alberta tokenizer
- Prepares datasets
- Trains using HuggingFace `Trainer`
- Evaluates on a held-out test set

```python
python train_sentiment_model.py
```

> **Note:** You can customize training parameters such as `batch_size`, `epochs`, and `learning_rate` inside the script.

---

## 🧪 Evaluation

After training, the model is evaluated using the following metrics:

- Accuracy
- F1 Score (weighted)
- Precision (weighted)
- Recall (weighted)

The best model (based on F1 score) is automatically saved and restored for evaluation.

---

## 🧠 Inference

Use the `classify_text` function to predict sentiment for new Hindi inputs:

```python
sample = "यह उत्पाद बहुत बेकार है।"
predicted_label = classify_text(sample)
print(f"Predicted Sentiment: {predicted_label}")
```

---

## 💾 Saving and Loading the Model

To save the fine-tuned model and tokenizer:

```python
model.save_pretrained("./albert-hindi-sentiment")
tokenizer.save_pretrained("./albert-hindi-sentiment")
```

To load it again later:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./albert-hindi-sentiment")
tokenizer = AutoTokenizer.from_pretrained("./albert-hindi-sentiment")
```

---
<!-- Test Results: {'eval_loss': 1.0780644416809082, 'eval_accuracy': 0.5924302788844622, 'eval_f1': 0.5607140730629933, 'eval_precision': 0.6457202644197677, 'eval_recall': 0.5924302788844622, 'eval_runtime': 19.3563, 'eval_samples_per_second': 129.673, 'eval_steps_per_second': 8.111, 'epoch': 45.0} -->


## 📊 Results

### Results on movies reviews dataset
| Metric    | Value |
|-----------|-------| 
| Accuracy  |  43.14|
| F1 Score  |  38.75|
| Precision |  42.18|
| Recall    |  43.14|

<!-- Test Results: {'eval_loss': 1.0733294486999512, 'eval_accuracy': 0.4314516129032258, 'eval_f1': 0.38750993231676073, 'eval_precision': 0.42177928312891294, 'eval_recall': 0.4314516129032258, 'eval_runtime': 20.5282, 'eval_samples_per_second': 72.486, 'eval_steps_per_second': 4.53, 'epoch': 36.0} -->

### Results on Product reviews dataset
| Metric    | Value |
|-----------|-------| 
| Accuracy  |  59.24|
| F1 Score  |  56.07|
| Precision |  64.57|
| Recall    |  59.243|

> You can update this table after training.


---

## 📌 Future Work

- Add support for class imbalance handling.
- Incorporate validation split for hyperparameter tuning.
- Integrate IndicNLP for preprocessing (normalization, sentence splitting).

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Albert](https://huggingface.co/albert/albert-base-v2)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)

---

## ✨ Contact

For queries, reach out to [pranitarora074@gmail.com] or create an issue.
