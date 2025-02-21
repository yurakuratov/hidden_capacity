import argparse

parser = argparse.ArgumentParser(description='Extract top N words from GloVe embeddings.')
parser.add_argument('--glove_path', type=str, help='Path to the GloVe embeddings file.')
parser.add_argument('--vocab_size', type=int, help='Number of words to extract.')
parser.add_argument('--output_path', type=str, help='Path to save the extracted embeddings.')


# https://nlp.stanford.edu/data/glove.6B.zip
# python make_vocab.py --glove_path ./glove.6B/glove.6B.50d.txt --vocab_size 100000 --output_path ./data/vocab_100k.txt
def main():
    args = parser.parse_args()

    glove_path = args.glove_path
    vocab_size = args.vocab_size
    output_path = args.output_path

    with open(glove_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= vocab_size:
                break
            word = line.split(' ')[0]
            outfile.write(f"{word}\n")

    print(f"Successfully extracted top {vocab_size} words to '{output_path}'.")


if __name__ == '__main__':
    main()
