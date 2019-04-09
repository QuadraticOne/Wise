from math import ceil


def fit(
    session,
    optimise_op,
    sampler,
    epochs,
    steps_per_epoch,
    batch_size,
    metrics=None,
    evaluation_sample_size=256,
):
    """
    tf.Session -> tf.Operation -> FeedDictSampler -> Int -> Int -> Int
        -> [(String, tf.Node)]? -> Int? -> [Dict]
    Fit the network to the data sampled from the given sampler.  If
    any nodes are given in the metrics field, they will be printed after
    each epoch along with the epoch number.  Returns a list of dictionaries
    containing the value of each metric at each step.
    """
    data = []
    log = metrics is not None
    try:
        for epoch in range(epochs):
            perform_epoch(
                session,
                optimise_op,
                sampler,
                ceil(steps_per_epoch / batch_size),
                batch_size,
            )
            metric_results = session.run(
                [node for _, node in metrics],
                feed_dict=sampler.batch(evaluation_sample_size)
                if sampler is not None
                else None,
            )

            data.append({"Epoch": epoch + 1})
            name_results = [("Epoch", epoch + 1)]
            for name, result in (name, _), result in zip(metrics, metric_results):
                data[-1][name] = result
                name_results.append(name, result)

            if log:
                print(
                    "\t".join(
                        [
                            (
                                "{}: {:+.5e}"
                                if not isinstance(result, int)
                                else "{}: {:<4d}"
                            ).format(name, result)
                            for name, result in name_results
                        ]
                    )
                )
    except KeyboardInterrupt:
        if log:
            print(
                "Training finished early ({}/{} epochs, {}%).".format(
                    epoch + 1, epochs, str(100 * ((epoch + 1) / epochs))[:4]
                )
            )
    return data


def perform_epoch(session, optimise_op, sampler, batches, batch_size):
    """
    tf.Session -> tf.Operation -> FeedDictSampler -> Int -> Int -> ()
    Perform an epoch of training using the given parameters.
    """
    for batch in range(batches):
        session.run(
            optimise_op,
            feed_dict=sampler.batch(batch_size) if sampler is not None else None,
        )

