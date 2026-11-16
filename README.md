# naive-bayes-from-scratch

A multinomial Naive Bayes text classifier, the evaluation metrics, and stratified
k-fold cross-validation, written in plain Python with **no ML libraries**.
The example task is detecting spam in Spanish SMS messages.

I wrote it to understand what libraries such as scikit-learn do internally:
where each probability comes from, why smoothing is needed, and how to evaluate
a classifier honestly on a small dataset.

## The maths

For a message with tokens $w_1, \dots, w_n$ the model picks the class

$$\hat{c} = \arg\max_c \; \log P(c) + \sum_{i=1}^{n} \log P(w_i \mid c)$$

assuming that words are independent given the class (the "naive" part).
Word likelihoods use Laplace (additive) smoothing, so a word that never
appeared in one class does not make its probability zero:

$$P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\text{total words}(c) + \alpha \lvert V \rvert}$$

Implementation choices:

- **Log-space**: multiplying hundreds of small probabilities underflows to `0.0`,
  so the model adds logarithms. Probabilities are recovered with the
  *log-sum-exp* trick (`predict_proba`). A test checks that a 5,000-word message
  still gets a valid probability.
- **Unknown words** are ignored at prediction time, so a message made only of
  new words falls back to the class prior.
- **Spanish-aware tokenizer**: lower-cases, removes accents but keeps `ñ`,
  drops very common words and maps every number to a single `<num>` token.

## Usage

Requires Python 3.10+ and no third-party packages.

```bash
pip install -e .
python -m nbayes evaluate data/sms_es.csv --folds 5
python -m nbayes predict data/sms_es.csv "Has ganado un premio, llama gratis ya"
```

```text
Dataset: data/sms_es.csv (60 messages, classes: ham, spam)
5-fold stratified cross-validation, alpha=1.0

Accuracy : 0.833 +/- 0.059
F1 (spam): 0.807 +/- 0.065

Confusion matrix (rows = true, columns = predicted)
         ham  spam
ham       29     7
spam       3    21

Most indicative words for 'spam': gratis, premio, tus, tarjeta, dinero, ganado, datos, aqui
```

As a library:

```python
from nbayes import MultinomialNaiveBayes

model = MultinomialNaiveBayes(alpha=1.0).fit(texts, labels)
model.predict("Reclama tu premio")          # 'spam'
model.predict_proba("¿Quedamos mañana?")    # {'ham': 0.9..., 'spam': 0.0...}
```

## Reading the results

- The dataset is **small (60 messages) and hand-written**. I made it to test the
  pipeline end to end, so the scores show that the code works, not how it would
  perform on real traffic.
- With only 12 test messages per fold, one mistake moves accuracy by about 8
  points. That is why the output shows the standard deviation across folds and
  uses **stratified** folds that keep the spam/ham ratio in every split.
- Most errors are **ham predicted as spam**. Spam messages are longer, so the
  spam class sees more words in training. Short everyday messages with words
  like "enlace" or "datos" can tip over. A larger dataset, or a threshold on
  `predict_proba`, would reduce false positives, which matter more than missed
  spam for this task.
- Changing `--alpha` from 1.0 to 0.1 improves accuracy slightly (0.85), but
  with this little data the difference is within the noise between folds.

## Project layout

```text
src/nbayes/
├── text.py      # normalisation and tokenisation
├── model.py     # MultinomialNaiveBayes
├── metrics.py   # accuracy, precision/recall/F1, confusion matrix
├── data.py      # CSV loading, stratified k-fold
└── __main__.py  # command-line interface
```

## Tests

```bash
pip install pytest
pytest
```

The tests check the smoothing formula against a hand computation, that
probabilities sum to 1, that long inputs do not underflow, the metrics on a
worked example (including the zero-division cases), and that the folds cover
every message exactly once while keeping the class balance.
