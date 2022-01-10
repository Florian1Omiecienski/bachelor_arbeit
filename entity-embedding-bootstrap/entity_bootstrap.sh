WORKING_DIR="./wikipedia_daten"
ENTITY_LIST="./debatenet_v2_daten/all_actor_wikidata_ids.txt "
FASTEXT_PATH="./data_fasttext_german/wiki.de.vec"
OUTPUT_DIR="./akteur_embeddings"

python3 "./entity-embedding-bootstrap/crawl.py" "$WORKING_DIR/" ba_official --reinit --entities $ENTITY_LIST
python3 "./entity-embedding-bootstrap/word_count.py"      "$WORKING_DIR/" $FASTEXT_PATH "$WORKING_DIR/global_word_count.list"
python3 "./entity-embedding-bootstrap/prepare.py"    "$WORKING_DIR/" ba_official
python3 "./entity-embedding-bootstrap/bootstrap.py" "$WORKING_DIR/entity_files/" $FASTEXT_PATH "$WORKING_DIR/global_word_count.list" "$OUTPUT_DIR/final_ba.embeddings.vec"

cat "$OUTPUT_DIR/final_ba.embeddings.vec" | awk '{print $1}' > "$WORKING_DIR/all_entities.list"
python3 ./entity-embedding-bootstrap/create_random_embeddings.py "$WORKING_DIR/all_entities.list" 300 "$OUTPUT_DIR/random_embeddings.vec"

