import copy
from collections import defaultdict


def replace_entity(input_ids_1, input_ids_2):
    input_ids_1 = copy.deepcopy(input_ids_1)
    input_ids_2 = copy.deepcopy(input_ids_2)
    subj_st1, subj_ed1 = input_ids_1.index(30522), input_ids_1.index(30523)
    obj_st1, obj_ed1 = input_ids_1.index(30524), input_ids_1.index(30525)
    subj_1 = input_ids_1[subj_st1 + 1: subj_ed1]
    obj_1 = input_ids_1[obj_st1 + 1: obj_ed1]

    subj_st2, subj_ed2 = input_ids_2.index(30522), input_ids_2.index(30523)
    obj_st2, obj_ed2 = input_ids_2.index(30524), input_ids_2.index(30525)

    ans_input_ids = []
    for idx in range(len(input_ids_2)):
        if subj_st2 < idx < subj_ed2 or obj_st2 < idx < obj_ed2:
            continue
        else:
            ans_input_ids.append(input_ids_2[idx])

    subj_sep_idx = ans_input_ids.index(30523)
    ans_input_ids = ans_input_ids[:subj_sep_idx] + subj_1 + ans_input_ids[subj_sep_idx:]
    obj_sep_idx = ans_input_ids.index(30525)
    ans_input_ids = ans_input_ids[:obj_sep_idx] + obj_1 + ans_input_ids[obj_sep_idx:]
    return ans_input_ids, [1] * len(ans_input_ids)


def remove_context(data):
    ans = []
    for idx in range(len(data)):
        ins = data[idx]
        input_ids = copy.deepcopy(ins["input_ids"])
        subj_st, subj_ed = input_ids.index(30522), input_ids.index(30523)
        obj_st, obj_ed = input_ids.index(30524), input_ids.index(30525)
        subj = input_ids[subj_st: subj_ed + 1]
        obj = input_ids[obj_st: obj_ed + 1]
        if subj_st < obj_st:
            input_ids = subj + obj
        else:
            input_ids = obj + subj
        subj_st, subj_ed = input_ids.index(30522), input_ids.index(30523)
        obj_st, obj_ed = input_ids.index(30524), input_ids.index(30525)
        ans.append({
            "input_ids": input_ids,
            "subject_marker_st": obj_st,
            "object_marker_st": subj_st,
            "labels": ins["labels"]
        })
    return ans


def relation_data_augmentation(data, num_labels, id2label, marker_id=(35022, 35023, 35024, 35025), augment_type="all"):
    subj_st_id, subj_ed_id, obj_st_id, obj_ed_id = marker_id
    new_label_dict = dict()

    add_reverse_relation = False
    add_undetermined_relation = False
    assert augment_type in ["all", "reverse", "no_rel", "none"]
    if augment_type == "reverse":
        add_reverse_relation = True
    elif augment_type == "no_rel":
        add_undetermined_relation = True
    elif augment_type == "all":
        add_undetermined_relation = True
        add_reverse_relation = True

    num_train_labels = num_labels

    if add_reverse_relation:
        # reverse relation augmentation for origin data
        augment_data = defaultdict(list)
        for ins in data:
            cur_label = ins["labels"]
            if id2label[cur_label] in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                                       'per:alternate_names', 'per:other_family']:
                continue
            if cur_label not in new_label_dict:
                new_label_dict[cur_label] = num_labels + len(new_label_dict)
            augment_label = new_label_dict[cur_label]
            input_ids = ins["input_ids"]
            subj_st, subj_ed = input_ids.index(subj_st_id), input_ids.index(subj_ed_id)
            obj_st, obj_ed = input_ids.index(obj_st_id), input_ids.index(obj_ed_id)
            aug_input_ids = copy.deepcopy(input_ids)
            aug_input_ids[subj_st] = obj_st_id
            aug_input_ids[subj_ed] = obj_ed_id
            aug_input_ids[obj_st] = subj_st_id
            aug_input_ids[obj_ed] = subj_ed_id
            augment_data[augment_label].append({
                "input_ids": aug_input_ids,
                "subject_marker_st": obj_st,
                "object_marker_st": subj_st,
                "labels": augment_label,
            })
        for _, v in augment_data.items():
            data.extend(v)
        num_train_labels += len(new_label_dict)

    if add_undetermined_relation:
        # undetermined relation augmentation for origin_data and augment_data
        for idx in range(len(data)):
            ins = data[idx]
            input_ids = copy.deepcopy(ins["input_ids"])
            subj_st, subj_ed = input_ids.index(subj_st_id), input_ids.index(subj_ed_id)
            obj_st, obj_ed = input_ids.index(obj_st_id), input_ids.index(obj_ed_id)
            subj = input_ids[subj_st: subj_ed + 1]
            obj = input_ids[obj_st: obj_ed + 1]
            if subj_st < obj_st:
                input_ids = subj + obj
            else:
                input_ids = obj + subj
            subj_st, subj_ed = input_ids.index(subj_st_id), input_ids.index(subj_ed_id)
            obj_st, obj_ed = input_ids.index(obj_st_id), input_ids.index(obj_ed_id)
            data.append({
                "input_ids": input_ids,
                "subject_marker_st": obj_st,
                "object_marker_st": subj_st,
                "labels": num_train_labels
            })
        num_train_labels += 1

    for idx in range(len(data)):
        for del_key in ["sentence", "input_ids_without_marker", "subject_st", "subject_ed", "object_st", "object_ed"]:
            if del_key in data[idx]:
                del data[idx][del_key]

    print(new_label_dict)
    return data, num_train_labels
