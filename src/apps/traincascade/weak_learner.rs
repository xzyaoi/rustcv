fn weak_learner(feature_collection: &mut Vec<HaarLikeFeature>,
                image_collection: &Vec<DataPoint>,
                weights: &Vector<f64>)
                -> HaarLikeFeature {
    let mut error_star = std::f64::INFINITY;
    let mut fi_star = None;
    let mut threshold_star = None;

    for fi in 0..(feature_collection.len()) {
        let ref mut feature_hypothesis = feature_collection[fi];

        let mut scores: Vec<_> = image_collection.iter()
            .enumerate()
            .map(|(index, ref data_point)| {
                (index, feature_hypothesis.get_score(&data_point.integral_image))
            })
            .collect();

        scores.sort_by(|&a, &b| a.partial_cmp(&b).unwrap());

        let mut error = image_collection.iter()
            .zip(weights.iter())
            .fold(0.0, |acc, (data_point, weight)| {
                if data_point.label * feature_hypothesis.polarity < 0.0 {
                    acc + weight
                } else {
                    acc
                }
            });

        let m = scores.len();
        for xi in 0..(m + 1) {
            let curr_x = if xi > 0 {
                scores[xi - 1].1
            } else {
                scores[xi].1 - 1.0
            };

            let next_x = if xi < m {
                scores[xi].1
            } else {
                scores[xi - 1].1 + 1.0
            };

            let threshold = (curr_x + next_x) / 2.0;
            feature_hypothesis.threshold = threshold;

            if error < error_star {
                error_star = error;
                fi_star = Some(fi);
                threshold_star = Some(threshold);
            }

            if xi < m {
                // computes the error of the next iteration
                let ref data_point = image_collection[xi];
                let weight = weights[xi];
                error += feature_hypothesis.polarity * data_point.label * weight;
            } else if error < error_star {
                // check after all error updates
                error_star = error;
                fi_star = Some(fi);
                threshold_star = Some(threshold);
            }
        }
    }

    let mut feature = feature_collection.remove(fi_star.unwrap());
    feature.threshold = threshold_star.unwrap();

    feature
}