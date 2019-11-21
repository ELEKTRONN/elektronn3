import numpy as np

def stats_overall_accuracy(cm):
    """Compute the overall accuracy.
    """
    return np.trace(cm)/cm.sum()


def stats_accuracy_per_class(cm):
    """Compute the accuracy per class and average, puts -1 for invalid values (division per 0),
        returns average accuracy, accuracy per class
    """
    # get number of predicted points per class
    sums = np.sum(cm, axis=1)
    mask = (sums > 0)

    # set to 1 to avoid division by zero
    sums[sums == 0] = 1
    accuracy_per_class = np.diag(cm) / sums

    # remove 1
    accuracy_per_class[np.logical_not(mask)] = -1

    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy, accuracy_per_class


def scores_from_predictions(predictions, category_list, label, category_range, data_num, label_test):
    shape_ious = {cat[0]: [] for cat in category_list}
    for shape_id, prediction in enumerate(predictions):
        segp = prediction
        cat = label[shape_id]
        category = category_list[cat][0]
        part_start, part_end = category_range[category]
        part_nbr = part_end - part_start
        point_num = data_num[shape_id]
        segl = label_test[shape_id][:point_num] - part_start

        part_ious = [0.0 for _ in range(part_nbr)]
        for l in range(part_nbr):
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l] = 1.0
            else:
                part_ious[l] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
        shape_ious[category].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    print(len(all_shape_ious))
    mean_shape_ious = np.mean(list(shape_ious.values()))
    for cat in sorted(shape_ious.keys()):
        print('eval mIoU of %s:\t %f' % (cat, shape_ious[cat]))
    print('eval mean mIoU: %f' % mean_shape_ious)
    print('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))