TRAINED_EMBEDDINGS="./akteur_embeddings/final_ba.embeddings.vec"
RANDOM_EMBEDDINGS="./akteur_embeddings/random_embeddings.vec"
FASTEXT_EMBEDDINGS="./data_fasttext_german/wiki.de.vec"
DEBATENET_DIR="./debatenet_v2_daten"
TRAINED_RESULT_DIR="./ergebnisse"
RANDOM_RESULT_DIR="./ergebnisse_random"


python3 ./modelle123/experiment.py $TRAINED_RESULT_DIR $TRAINED_EMBEDDINGS $FASTEXT_EMBEDDINGS $DEBATENET_DIR 
python3 ./modelle123/evaluate.py    "$TRAINED_RESULT_DIR" "$TRAINED_EMBEDDINGS" "$DEBATENET_DIR"


python3 ./modelle123/experiment.py $RANDOM_RESULT_DIR $RANDOM_EMBEDDINGS $FASTEXT_EMBEDDINGS $DEBATENET_DIR 
python3 ./modelle123/evaluate.py   "$RANDOM_RESULT_DIR" "$RANDOM_EMBEDDINGS" "$DEBATENET_DIR"

python3 ./modelle123/show_results.py "$TRAINED_RESULT_DIR/evaluation_results.bin"
python3 ./modelle123/show_results.py "$RANDOM_RESULT_DIR/evaluation_results.bin"

