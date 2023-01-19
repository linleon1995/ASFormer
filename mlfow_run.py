import mlflow


def main():
    with mlflow.start_run(run_id='test', experiment_id='VAS'):
        pass


if __name__ == '__main__':
    main()
