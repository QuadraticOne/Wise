from math import ceil


def fit(session, optimise_op, sampler, epochs, steps_per_epoch, batch_size,
        metrics=None, evaluation_sample_size=256):
    """
    tf.Session -> tf.Operation -> FeedDictSampler -> Int -> Int -> Int
        -> [(String, tf.Node)]? -> Int? -> ()
    Fit the network to the data sampled from the given sampler.  If
    any nodes are given in the metrics field, they will be printed after
    each epoch along with the epoch number.
    """
    log = metrics is not None
    try:
        for epoch in range(epochs):
            perform_epoch(session, optimise_op, sampler,
                ceil(steps_per_epoch / batch_size), batch_size)
            if log:
                metric_results = session.run([node for _, node in metrics],
                    feed_dict=sampler.batch(evaluation_sample_size))
                name_results = [(name, result) \
                    for (name, _), result in zip(metrics, metric_results)]
                name_results.insert(0, ('Epoch', epoch))
                print('\t'.join(['{}: {}'.format(name, result) \
                    for name, result in name_results]))
    except KeyboardInterrupt:
        if log:
            print('Training finished early ({}/{} epochs, {}%).'
                .format(epoch, epochs, str(100 * (epoch / epochs))[:4]))


def perform_epoch(session, optimise_op, sampler, batches, batch_size):
    """
    tf.Session -> tf.Operation -> FeedDictSampler -> Int -> Int -> ()
    Perform an epoch of training using the given parameters.
    """
    for batch in range(batches):
        session.run(optimise_op, feed_dict=sampler.batch(batch_size))

