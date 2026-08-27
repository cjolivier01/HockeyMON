// #include "BYTETracker.h"
// #include "Lapjv.h"
// #include "STrack.h"

#include <iostream>
#include <map>

namespace hm {
namespace tracker {
#if 0
vector<OldSTrack*> BYTETracker::joint_stracks(
    vector<OldSTrack*>& tlista,
    vector<OldSTrack>& tlistb) {
  map<int, int> exists;
  vector<OldSTrack*> res;
  for (int i = 0; i < tlista.size(); i++) {
    exists.insert(pair<int, int>(tlista[i]->track_id, 1));
    res.push_back(tlista[i]);
  }
  for (int i = 0; i < tlistb.size(); i++) {
    int tid = tlistb[i].track_id;
    if (!exists[tid] || exists.count(tid) == 0) {
      exists[tid] = 1;
      res.push_back(&tlistb[i]);
    }
  }
  return res;
}

vector<OldSTrack> BYTETracker::joint_stracks(
    vector<OldSTrack>& tlista,
    vector<OldSTrack>& tlistb) {
  std::map<int, int> exists;
  vector<OldSTrack> res;
  for (int i = 0; i < tlista.size(); i++) {
    exists.insert(pair<int, int>(tlista[i].track_id, 1));
    res.push_back(tlista[i]);
  }
  for (int i = 0; i < tlistb.size(); i++) {
    int tid = tlistb[i].track_id;
    if (!exists[tid] || exists.count(tid) == 0) {
      exists[tid] = 1;
      res.push_back(tlistb[i]);
    }
  }
  return res;
}

vector<OldSTrack> BYTETracker::sub_stracks(
    vector<OldSTrack>& tlista,
    vector<OldSTrack>& tlistb) {
  map<int, OldSTrack> stracks;
  for (int i = 0; i < tlista.size(); i++) {
    stracks.insert(pair<int, OldSTrack>(tlista[i].track_id, tlista[i]));
  }
  for (int i = 0; i < tlistb.size(); i++) {
    int tid = tlistb[i].track_id;
    if (stracks.count(tid) != 0) {
      stracks.erase(tid);
    }
  }

  vector<OldSTrack> res;
  std::map<int, OldSTrack>::iterator it;
  for (it = stracks.begin(); it != stracks.end(); ++it) {
    res.push_back(it->second);
  }

  return res;
}

void BYTETracker::remove_duplicate_stracks(
    vector<OldSTrack>& resa,
    vector<OldSTrack>& resb,
    vector<OldSTrack>& stracksa,
    vector<OldSTrack>& stracksb) {
  vector<vector<float>> pdist = iou_distance(stracksa, stracksb);
  vector<pair<int, int>> pairs;
  for (int i = 0; i < pdist.size(); i++) {
    for (int j = 0; j < pdist[i].size(); j++) {
      if (pdist[i][j] < 0.15) {
        pairs.push_back(pair<int, int>(i, j));
      }
    }
  }

  vector<int> dupa, dupb;
  for (int i = 0; i < pairs.size(); i++) {
    int timep = stracksa[pairs[i].first].frame_id -
        stracksa[pairs[i].first].start_frame;
    int timeq = stracksb[pairs[i].second].frame_id -
        stracksb[pairs[i].second].start_frame;
    if (timep > timeq)
      dupb.push_back(pairs[i].second);
    else
      dupa.push_back(pairs[i].first);
  }

  for (int i = 0; i < stracksa.size(); i++) {
    vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
    if (iter == dupa.end()) {
      resa.push_back(stracksa[i]);
    }
  }

  for (int i = 0; i < stracksb.size(); i++) {
    vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
    if (iter == dupb.end()) {
      resb.push_back(stracksb[i]);
    }
  }
}

void BYTETracker::linear_assignment(
    vector<vector<float>>& cost_matrix,
    int cost_matrix_size,
    int cost_matrix_size_size,
    float thresh,
    vector<vector<int>>& matches,
    vector<int>& unmatched_a,
    vector<int>& unmatched_b) {
  if (cost_matrix.size() == 0) {
    for (int i = 0; i < cost_matrix_size; i++) {
      unmatched_a.push_back(i);
    }
    for (int i = 0; i < cost_matrix_size_size; i++) {
      unmatched_b.push_back(i);
    }
    return;
  }

  vector<int> rowsol;
  vector<int> colsol;
  float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
  for (int i = 0; i < rowsol.size(); i++) {
    if (rowsol[i] >= 0) {
      vector<int> match;
      match.push_back(i);
      match.push_back(rowsol[i]);
      matches.push_back(match);
    } else {
      unmatched_a.push_back(i);
    }
  }

  for (int i = 0; i < colsol.size(); i++) {
    if (colsol[i] < 0) {
      unmatched_b.push_back(i);
    }
  }
}

vector<vector<float>> BYTETracker::ious(
    vector<vector<float>>& atlbrs,
    vector<vector<float>>& btlbrs) {
  vector<vector<float>> ious;
  if (atlbrs.size() * btlbrs.size() == 0)
    return ious;

  ious.resize(atlbrs.size());
  for (int i = 0; i < ious.size(); i++) {
    ious[i].resize(btlbrs.size());
  }

  // bbox_ious
  for (int k = 0; k < btlbrs.size(); k++) {
    vector<float> ious_tmp;
    float box_area =
        (btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
    for (int n = 0; n < atlbrs.size(); n++) {
      float iw =
          min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
      if (iw > 0) {
        float ih = min(atlbrs[n][3], btlbrs[k][3]) -
            max(atlbrs[n][1], btlbrs[k][1]) + 1;
        if (ih > 0) {
          float ua = (atlbrs[n][2] - atlbrs[n][0] + 1) *
                  (atlbrs[n][3] - atlbrs[n][1] + 1) +
              box_area - iw * ih;
          ious[n][k] = iw * ih / ua;
        } else {
          ious[n][k] = 0.0;
        }
      } else {
        ious[n][k] = 0.0;
      }
    }
  }

  return ious;
}

vector<vector<float>> BYTETracker::iou_distance(
    vector<OldSTrack*>& atracks,
    vector<OldSTrack>& btracks,
    int& dist_size,
    int& dist_size_size) {
  vector<vector<float>> cost_matrix;
  if (atracks.size() * btracks.size() == 0) {
    dist_size = atracks.size();
    dist_size_size = btracks.size();
    return cost_matrix;
  }
  vector<vector<float>> atlbrs, btlbrs;
  for (int i = 0; i < atracks.size(); i++) {
    atlbrs.push_back(atracks[i]->tlbr);
  }
  for (int i = 0; i < btracks.size(); i++) {
    btlbrs.push_back(btracks[i].tlbr);
  }

  dist_size = atracks.size();
  dist_size_size = btracks.size();

  vector<vector<float>> _ious = ious(atlbrs, btlbrs);

  for (int i = 0; i < _ious.size(); i++) {
    vector<float> _iou;
    for (int j = 0; j < _ious[i].size(); j++) {
      _iou.push_back(1 - _ious[i][j]);
    }
    cost_matrix.push_back(_iou);
  }

  return cost_matrix;
}

vector<vector<float>> BYTETracker::iou_distance(
    vector<OldSTrack>& atracks,
    vector<OldSTrack>& btracks) {
  vector<vector<float>> atlbrs, btlbrs;
  for (int i = 0; i < atracks.size(); i++) {
    atlbrs.push_back(atracks[i].tlbr);
  }
  for (int i = 0; i < btracks.size(); i++) {
    btlbrs.push_back(btracks[i].tlbr);
  }

  vector<vector<float>> _ious = ious(atlbrs, btlbrs);
  vector<vector<float>> cost_matrix;
  for (int i = 0; i < _ious.size(); i++) {
    vector<float> _iou;
    for (int j = 0; j < _ious[i].size(); j++) {
      _iou.push_back(1 - _ious[i][j]);
    }
    cost_matrix.push_back(_iou);
  }

  return cost_matrix;
}
#endif

} // namespace tracker
} // namespace hm
