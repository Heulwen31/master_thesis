from river import forest, tree, ensemble

class RiverModelFactory:
    @staticmethod
    def get_model(algorithm, **kwargs):
        """
        Returns an initialized River model instance.
        Supported kwargs: n_models, lambda_value, grace_period,
        split_criterion, leaf_prediction, max_depth, seed, etc.
        """
        n_models = kwargs.get("n_models", 10)
        lambda_value = kwargs.get("lambda_value", 6)
        grace_period = kwargs.get("grace_period", 200)
        split_criterion = kwargs.get("split_criterion", "gini")
        leaf_prediction = kwargs.get("leaf_prediction", "nba")
        max_depth = kwargs.get("max_depth", None)
        seed = kwargs.get("seed", 42)

        if algorithm == "ARF":
            return forest.ARFClassifier(
                n_models=n_models,
                lambda_value=lambda_value,
                grace_period=grace_period,
                split_criterion=split_criterion,
                leaf_prediction=leaf_prediction,
                max_depth=max_depth,
                seed=seed,
            )
        elif algorithm == "HAT":
            return tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=grace_period,
                split_criterion=split_criterion,
                leaf_prediction=leaf_prediction,
                max_depth=max_depth,
                seed=seed,
            )
        elif algorithm == "LeveragingBagging":
            return ensemble.LeveragingBaggingClassifier(
                model=tree.HoeffdingTreeClassifier(
                    split_criterion=split_criterion,
                    leaf_prediction=leaf_prediction,
                    max_depth=max_depth,
                    grace_period=grace_period,
                ),
                n_models=n_models,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown River algorithm: {algorithm}")
