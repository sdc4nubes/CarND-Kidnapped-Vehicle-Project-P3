/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles
	num_particles = 300;
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);
	// Initialize all particles to first position and set all weights to 1
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x  = x;
		p.y  = y;
		p.theta = theta;
		p.weight = 1.0;
		// Add random Gaussian noise to each particle
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
		particles.push_back(p);
		weights.push_back(p.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity,
                                double yaw_rate) {
	// Initialize noise distribution
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	// Add measurements to each particle
	for (int i = 0; i < num_particles; i++) {
		double new_x, new_y, new_theta;
		if(yaw_rate == 0){
			new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		} else {
			new_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta +
                    yaw_rate * delta_t) - sin(particles[i].theta));
        	new_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) -
                    cos(particles[i].theta + yaw_rate*delta_t));
        	new_theta = particles[i].theta + yaw_rate*delta_t;
		}
		// Add random Gaussian noise to each particle
		particles[i].x = new_x + dist_x(gen);
		particles[i].y = new_y + dist_y(gen);
		particles[i].theta = new_theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement
	for (int i = 0; i < observations.size(); i++) {
		LandmarkObs observation = observations[i];
		// Initialize the landmark id (-1) and minimum distance (maximum possible distance)
		int landmark_id = -1;
		double min_distance = numeric_limits<double>::max();
		// Find the landmark with the minimum distance
		for(int j = 0; j < predicted.size(); j++){
      LandmarkObs predicted_measurement = predicted[j];
      double distance = dist(observation.x, observation.y, predicted_measurement.x,
                              predicted_measurement.y);
      if (distance < min_distance){
        min_distance = distance;
        landmark_id = predicted_measurement.id;
      }
    }
    // Assign the landmark id with the minimum distance to the observed measurement
    observations[i].id = landmark_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	for (int i = 0; i < num_particles; i++){
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		// Store predicted landmark locations that are inside the particle sensor range
    vector<LandmarkObs> predicted_landmarks;
    for (int k = 0; k < map_landmarks.landmark_list.size(); k++){
    	int lm_id = map_landmarks.landmark_list[k].id_i;
    	double lm_x = map_landmarks.landmark_list[k].x_f;
    	double lm_y = map_landmarks.landmark_list[k].y_f;
    	LandmarkObs curr_lm = {lm_id, lm_x, lm_y};
    	// Choose landmarks in the sensor range of the particle
    	if (fabs(dist(lm_x, lm_y, x, y)) <= sensor_range){
        	predicted_landmarks.push_back(curr_lm);
      }
    }
    // Transform observations from vehicle's coordinates to map's coordinate
    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations.size(); j++){
    	// Apply rotation and translation
    	double tr_x = observations[j].x * cos(theta) - observations[j].y *
                        sin(theta) + x;
    	double tr_y = observations[j].x * sin(theta) + observations[j].y *
                        cos(theta) + y;
    	LandmarkObs transformed_ob;
    	transformed_ob.id = observations[j].id;
    	transformed_ob.x = tr_x;
    	transformed_ob.y = tr_y;
    	transformed_observations.push_back(transformed_ob);
    }
    dataAssociation(predicted_landmarks, transformed_observations);
    // Update weights
    double total_weight = 1.0;
    weights[i] = 1.0;
    for (int a = 0; a < transformed_observations.size(); a++){
    	int obv_id = transformed_observations[a].id;
    	double obv_x = transformed_observations[a].x;
    	double obv_y = transformed_observations[a].y;
    	double pred_x;
    	double pred_y;
    	for (int b = 0; b < predicted_landmarks.size(); b++){
    		if (predicted_landmarks[b].id == obv_id){
    			pred_x = predicted_landmarks[b].x;
    			pred_y = predicted_landmarks[b].y;
    		}
    	}
    	// Apply multivariate Gaussian distribution
    	double w = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) *
                      exp(-(pow(pred_x - obv_x, 2) / (2 * pow(std_landmark[0],2)) +
                      pow(pred_y - obv_y, 2)/(2 * pow(std_landmark[1],2)) -
                      (2 * (pred_x - obv_x) * (pred_y - obv_y)/(sqrt(std_landmark[0]) *
                      sqrt(std_landmark[1])))));
      total_weight *= w;
    }
    particles[i].weight = total_weight;
    weights[i] = total_weight;
    predicted_landmarks.clear();
	}
}

void ParticleFilter::resample() {
	// Create particle distribution with probabilities proportional to weights
	discrete_distribution<int> index(weights.begin(), weights.end());
	// Initialize resampled particles vector
	vector<Particle> resampled_particles;
	// Resample particles with replacement with probability proportional to weights
	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[index(gen)]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  // Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
	// Assign particles to associations and world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}