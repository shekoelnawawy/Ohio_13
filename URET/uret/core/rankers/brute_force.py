from uret.core.rankers.ranking_algorithm import RankingAlgorithm
import warnings
import copy


class BruteForce(RankingAlgorithm):
    """
    This implementation tries all transformations and parameters for each given sample,
    and returns the list of them with scores and applied transformations.
    """

    def __init__(self, transformer_list, multi_feature_input=False, is_trained=True):
        """
        :param transformer_list: List of available transformations to be used by the algorithm.
        :param multi_feature_input: A boolean flag indicating if each index in the transformer
            list defines a transformer(s) for a single data type in the input. If False, then it
            is assumed that the transformer list defines multiple transformers for a single input
        :param is_trained: Indicates when _train has been called at least once. Always True
        """
        super().__init__(transformer_list, multi_feature_input, is_trained=True)

    # Nawawy's start
    def rank_edges(self, sample, scoring_function, score_input, model_predict, feature_extractor, dependencies=[],
                   current_transformation_records=None):
        backcast = sample[1]
        nv = sample[2]
        sample = sample[0]
    # Nawawy's end

        # Create transformation record
        if self.multi_feature_input and current_transformation_records is None:
            current_transformation_records = [None for _ in range(len(self.transformer_list))]

        return_values = []  # This will contain the (sample_index, transformer, action args, transformed sample,
        # transformation_record of the transformed sample, score)

        for transformer_index, (transformer, input_index) in enumerate(self.transformer_list):
            if self.multi_feature_input:
                possible_actions = transformer.get_possible(
                    sample[input_index], transformation_record=current_transformation_records[transformer_index]
                )
            else:
                possible_actions = transformer.get_possible(
                    sample, transformation_record=current_transformation_records
                )
            for action in possible_actions:
                sample_temp = copy.copy(sample)
                transformation_records_temp = copy.deepcopy(current_transformation_records)

                if self.multi_feature_input:
                    transformed_value, new_transformation_record = transformer.transform(
                        sample_temp[input_index],
                        transformation_record=transformation_records_temp[transformer_index],
                        transformation_value=action,
                    )
                    sample_temp[input_index] = transformed_value
                    transformation_records_temp[transformer_index] = new_transformation_record
                else:
                    transformed_value, new_transformation_record = transformer.transform(
                        sample_temp, transformation_record=transformation_records_temp, transformation_value=action
                    )
                    sample_temp = transformed_value
                    transformation_records_temp = new_transformation_record

                sample_temp = self._enforce_dependencies(sample_temp, dependencies)
                # Nawawy's start
                sample_temp = sample_temp.reshape(1, backcast, nv)
                new_prediction, _, _, _, _ = model_predict(feature_extractor(sample_temp))
                score = scoring_function(new_prediction, score_input)

                sample_temp = sample_temp.reshape(backcast * nv)
                # Nawawy's end
                if self.multi_feature_input:
                    return_values.append(
                        (
                            [transformer_index, input_index],
                            transformer,
                            action,
                            sample_temp,
                            transformation_records_temp,
                            score,
                        )
                    )
                else:
                    return_values.append([None, transformer, action, sample_temp, transformation_records_temp, score])
        return return_values
