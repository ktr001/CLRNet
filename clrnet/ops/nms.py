# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch


def nms(boxes, scores, overlap, top_k):
    """Python fallback lane NMS (replaces CUDA nms_impl).

    CLRNet's CUDA NMS suppresses lanes whose x-coordinate overlap exceeds
    `overlap`. This approximation sorts by score and greedily suppresses lanes
    whose mean |Δx| over valid points is below `overlap` pixels.

    Returns: (kept_indices, num_to_keep, None)  – 3-element tuple matching
    the original CUDA return signature.
    """
    if boxes.shape[0] == 0:
        empty = torch.empty((0,), dtype=torch.long, device=boxes.device)
        return empty, 0, None

    # Sort by confidence (scores already sorted upstream, but sort anyway)
    order = scores.argsort(descending=True)
    boxes = boxes[order]

    n = boxes.shape[0]
    # Lane x positions are in columns 5: (after start_y, end_y, start_x,
    # end_x, length in CLRNet's internal representation)
    lane_xs = boxes[:, 5:].float()  # (N, n_points)
    valid = lane_xs > 0             # (N, n_points)

    suppressed = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    kept_list = []

    for i in range(n):
        if suppressed[i]:
            continue
        kept_list.append(i)
        if len(kept_list) >= top_k:
            break
        # Suppress lanes that are too similar to lane i
        for j in range(i + 1, n):
            if suppressed[j]:
                continue
            v = valid[i] & valid[j]
            if v.sum() == 0:
                continue
            mean_dist = (lane_xs[i][v] - lane_xs[j][v]).abs().mean()
            if mean_dist < overlap:
                suppressed[j] = True

    if not kept_list:
        empty = torch.empty((0,), dtype=torch.long, device=boxes.device)
        return empty, 0, None

    kept = order[torch.tensor(kept_list, dtype=torch.long, device=boxes.device)]
    return kept, len(kept_list), None
