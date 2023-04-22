import sys
import argparse
import ir_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    dataset = ir_datasets.load(args.dataset)
    qids = {q.query_id for q in dataset.queries}
    for line in sys.stdin:
        qid = line.split()[0]
        if qid in qids:
            sys.stdout.write(line)

if __name__ == '__main__':
    main()
