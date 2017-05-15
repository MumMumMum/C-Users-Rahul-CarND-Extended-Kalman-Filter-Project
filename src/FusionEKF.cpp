#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2); // (p , v)px,py,vx,vy
    R_radar_ = MatrixXd(3, 3); //row,phi,row_,(px,py,vx,vy)
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;

    /**
    TODO:
      * Finish initializing the FusionEKF.
      * Set the process and measurement noises
    */
    //H is measurement Matrix is linear for Laser
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;
    //H is measurement Matrix is non linear for Radar
    Hj_ <<  1, 1, 0, 0,
            1, 1, 0, 0,
            1, 1, 1, 1;

    // Initalize state Covariance Matric P
    //As Prediction is linear in Laser and Radar type filter we have only P matrix

    //  [.11, .11, 0.52, 0.52].
    ekf_.P_ = MatrixXd(4, 4);
    // Initalize state Covariance Matric P for radar
    ekf_.P_ << 1, 0, 0, 0, //px
               0, 1, 0, 0,   //py
               0, 0, 2, 0, //vx
               0, 0, 0, 10; //vy


    //Initalize the inital state transition Matric
    ekf_.F_  = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;
    //Initallize the Qmatrix, This helps us adding the process
    //Noise, accelatration in speed to the filter
    ekf_.Q_  = MatrixXd(4, 4);
    ekf_.Q_ << 0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0;
    //Measurement noise is 0 for L type sensors.
    //set the acceleration noise components
    noise_ax = 9; // process noise in x direction
    noise_ay = 9; // process noise in y direction


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
        TODO:
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // first measurement
        cout << "EKF: " << endl;

        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float radialDistance = measurement_pack.raw_measurements_[0];
            float anglePhi = measurement_pack.raw_measurements_[1];
            float radialValocity = measurement_pack.raw_measurements_[2];
            ekf_.x_(0) = radialDistance * cos(anglePhi);
            ekf_.x_(1) = radialDistance * sin(anglePhi);
            ekf_.x_(2) = radialValocity * cos(anglePhi);
            ekf_.x_(3) = radialValocity * sin(anglePhi);


        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }

        // done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
     TODO:
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    //compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;    //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    //1. Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;
    //2. Set the process covariance matrix Q
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;


    ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
            0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
            dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
            0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;


    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     TODO:
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        // As the input data is non linear,
        // we find Jacobian matrix and then update the kalman filter
        Tools tools;
        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_  = Eigen::MatrixXd(3, 4);
        ekf_.H_ = Hj_;
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);

    } else {
        // Laser updates
        // As the ip data  linear we dirctly update the kalman filter
        ekf_.H_  = Eigen::MatrixXd(2, 4);
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}


