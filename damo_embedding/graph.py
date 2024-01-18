#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright (C) 2019 - present timepi <timepi123@gmail.com>
# `Damo-Embedding` is provided under: GNU Affero General Public License
# (AGPL3.0) https:#www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#

# this file extracts sparse parts from model and updates the graph

import json
from typing import Dict, List, Tuple

import numpy as np
import torch

GLOBAL_SPARSE_EMBEDDING_USE_COUNT = 0


def is_node_for_sparse_embedding(node: torch.Node, input_node: torch.Value) -> bool:
    """is a node for a sparse embedding

    Args:
        node (torch.Node): node
        input_node (torch.Value): input node

    Returns:
        bool: is sparse embedding or not
    """
    SPARSE_TYPENAME_SUFFIX = "DummyEmbedding"
    inputs = list(node.inputs())
    if len(inputs) != 2:
        return False
    if inputs[1] != input_node:
        return False
    ntype = str(inputs[0].type())
    if not ntype.endswith(SPARSE_TYPENAME_SUFFIX):
        return False
    return True


def extract_attribute_key(node: torch.Node) -> str:
    """extract attribute key

    Args:
        node (torch.Node): sparse node to extract attribute key

    Returns:
        str: model attribute key
    """
    global GLOBAL_SPARSE_EMBEDDING_USE_COUNT
    GLOBAL_SPARSE_EMBEDDING_USE_COUNT += 1
    inputs = list(node.inputs())
    n = inputs[0].node()
    assert n.kind() == "prim::GetAttr"
    n_inputs = list(n.inputs())
    assert len(n_inputs) == 1
    assert n_inputs[0].debugName().startswith("self")
    origin = n.s("name")
    name = f"{origin}__{GLOBAL_SPARSE_EMBEDDING_USE_COUNT}"
    n.s_("name", name)
    return origin


def update_sparse_parts_from_graph(model: torch.nn.Module) -> List:
    """update graph

    Args:
        model (torch.nn.Module): model to update

    Returns:
        List: inputs process list
    """
    # get model's sparse embeddings
    embeddings = get_sparse_embeddings(model)
    inputs = list(model.graph.inputs())[1:]
    ret = []
    matches = []
    for i, node in enumerate(inputs):
        users = list(map(lambda x: x.user, node.uses()))
        if len(users) == 0:
            continue
        status = is_node_for_sparse_embedding(users[0], node)

        # if input i for sparse embedding,
        # then all nodes use this input must be sparse embedding.
        for user in users:
            s = is_node_for_sparse_embedding(user, node)
            assert s == status
        if not status:
            continue
        start = 0
        keys, dims = [], []

        for j, user in enumerate(users):
            key = extract_attribute_key(user)
            dim = embeddings[key]["dim"]
            keys.append(key)
            dims.append(dim)
            pattern, replacement = build_pattern_replacement_graph(
                user, node, start, dim
            )
            start += dim
            matches.append((pattern, replacement))
        ret.append({"input": i, "keys": keys, "dims": dims})
    for pattern, replacement in matches:
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            pattern, replacement, model.graph
        )
    return ret


def get_sparse_embeddings(model: torch.nn.Module) -> Dict[str, Dict[str, np.ndarray]]:
    """get sparse embedding from model
    currently only supports sparse embedding as model's attributes

    Args:
        model (torch.nn.Module): torch model object

    Returns:
        Dict[str, Dict[str, np.ndarray]]: embeddings
    """
    embeddings = {}
    for k, v in model.__dict__["_modules"].items():
        if v.original_name == "DummyEmbedding":
            dim = v.embedding.weight.shape[1]
            embeddings[k] = {
                "dim": dim,
            }
    return embeddings


def build_pattern_replacement_graph(
    node: torch.Node, input_node: torch.Value, start: int, dim: int
) -> Tuple[str, str]:
    """build the replacement pattern

    Args:
        node (torch.Node): node
        input_node (torch.Value): input
        start (int): slice starting
        dim (int): slice width

    Returns:
        Tuple[str,str]: pattern and replacement
    """

    def remove_comments(code: str) -> str:
        idx = code.find("#")
        if idx >= 0:
            code = code[:idx]
        return code.strip()

    prev = node.prev()
    prev_attr = prev.s("name")
    prev_input = f"%{list(prev.inputs())[0].debugName()}"
    prev_output = f"%{list(prev.outputs())[0].debugName()}"

    input = f"%{input_node.debugName()}"
    output = f"%{list(node.outputs())[0].debugName()}"
    code_0 = f'{prev_output} = prim::GetAttr[name="{prev_attr}"]({prev_input})'
    code_1 = remove_comments(str(node))

    pattern = f"""
    graph({prev_input}, {input}): 
        {code_0}
        {code_1}
        return ({output}) 
    """
    end = start + dim
    replacement = f"""
    graph({prev_input}, {input}): 
        %1 : NoneType = prim::Constant()
        %2 : int = prim::Constant[value=-1]()
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value={start}]()
        %5 : int = prim::Constant[value={end}]()
        %out : Tensor= aten::slice({input}, %2, %4, %5, %3) 
        return (%out)
    """
    return pattern, replacement


def update_model_graph(model: torch.nn.Module, model_path: str, meta_path: str):
    """update model graph

    Args:
        model (torch.nn.Module): original model
        model_path (str): new model path
        meta_path (str): meta file path
    """
    meta = update_sparse_parts_from_graph(model)
    json.dump({"meta": meta, "sparse": 1}, open(meta_path, "w"))
    model.save(model_path)


def update_model(original_model_path: str, model_path: str, meta_path: str):
    """update model graph

    Args:
        model (str): original model
        model_path (str): new model path
        meta_path (str): meta file path
    """
    model = torch.jit.load(original_model_path)
    update_model_graph(model=model, model_path=model_path, meta_path=meta_path)
