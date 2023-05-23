def preprocess(data, unmodified_indices = None):
    """
    A function to control transforming input values where parts of the input should remain unchanged.
    :param data: Patients' data.
    :param unmodified_indices: Feature indices that should not be modified.
    :return: modifiable sections, unmodifiable sections
    """
    if unmodified_indices is None:
        return data, []
    else:

        all_indices = list(range(0, data[0].shape[1]))
        modified_indices = list(set(all_indices) - set(unmodified_indices))
        modifiable_list = []
        unmodifiable_list = []
        for i in range(len(data)):
            patient_modifiable = data[i][:, modified_indices]
            patient_unmodifiable = data[i][:, unmodified_indices]
            modifiable_list.append(patient_modifiable)
            unmodifiable_list.append(patient_unmodifiable)

        return modifiable_list, unmodifiable_list
