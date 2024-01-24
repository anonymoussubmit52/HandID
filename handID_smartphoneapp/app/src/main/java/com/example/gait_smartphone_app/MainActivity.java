package com.example.gait_smartphone_app;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    /**
     * This is for data collection
     * >> dataEarth, dataDevice are the raw data String which is convenient to be wrote into CSV
     */
    // data (device acc data, earth acc data, gyroscope data)
    public static StringBuilder acc_data = new StringBuilder();
    public static StringBuilder gyro_data = new StringBuilder();
    public static StringBuilder heart_data = new StringBuilder();

    // frequency - 100Hz
    public static int FREQUENCY_100 = 2500;

    private volatile boolean start = false;
    private volatile String accOutputFileName = "";
    private volatile String gyroOutputFileName = "";
    private volatile String heartOutputFileName = "";

    private int INTERVAL = 0;

    @SuppressLint("SimpleDateFormat")
    private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private TextView startButton = null;

    // low pass filter
    private final float[] rawAccData = new float[3];
    private final float[] LPFAccData = new float[3];
    private final float[] LPFPreAccData = new float[3];
    private int count = 0;
    private float beginTime;
    private float rc = 0.1f;
    private int repeat = 10;

    // gravity sensor data
    private float[] gravityValues = null;
    private float[] magneticValues = null;

    int VIBRATION_DURATION = 3;
    int VIBRATION_DELAY = 1;
    int VIBRATION_EXTENSION = 1;

    private volatile boolean saved = false;
    private int times = 0;
    private String pre = "gesture";


    // ==================================================
    // vibrator
    Vibrator myVibrator;
    public EditText gesture_order;

    @SuppressLint("WrongConstant")
    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        SensorManager sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        Sensor sensor_acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Sensor sensor_gra = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        Sensor sensor_mag = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        Sensor sensor_gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Sensor sensor_ppg = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE);
        // Find the frequency with 100Hz
        sensorManager.registerListener(this, sensor_acc, FREQUENCY_100);
        sensorManager.registerListener(this, sensor_gyro, FREQUENCY_100);
        sensorManager.registerListener(this, sensor_gra, FREQUENCY_100);
        sensorManager.registerListener(this, sensor_mag, FREQUENCY_100);
        sensorManager.registerListener(this, sensor_ppg, FREQUENCY_100);

        INTERVAL = VIBRATION_DELAY*1000 + VIBRATION_DURATION*1000 + VIBRATION_EXTENSION * 1000;
        gesture_order = findViewById(R.id.editTextNumber);

        this.startButton = findViewById(R.id.start_button);
        this.startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                while (gesture_order.getText().toString().trim().equals("")) {

                }
                gesture_order.setVisibility(View.GONE);

                pre = pre + gesture_order.getText().toString().trim() + "_";
                OutputDataThread outputDataThread = new OutputDataThread();
                outputDataThread.setDaemon(true);
                outputDataThread.start();

                start_cp();
            }
        });

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                    && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                System.out.println(">>>>> HAVE PERMISSIONS <<<<<");
            } else {
                ActivityCompat.requestPermissions(this, new String[]
                        {
                                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                                Manifest.permission.READ_EXTERNAL_STORAGE,
                                Manifest.permission.BODY_SENSORS
                        }, 2);
                System.out.println(">>>>> PERMISSION GRANTED <<<<<");
            }
        }

        myVibrator = (Vibrator) getSystemService(Service.VIBRATOR_SERVICE);
    }


    private Map<String, Object> accToAmplitudeMap(ArrayList<Float> original_y) {
        Map<String, Object> ret = new HashMap<>();
        int[] amp = new int[original_y.size()+1];
        long[] duration = new long[original_y.size()+1];
        amp[0] = 0;
        duration[0] = 0;

        // Try -8 to +8, 32 interval
        for (int i = 0; i < 30; i++) {
            amp[i+1] = (int) Math.abs(original_y.get(i)) * 32;
            duration[i+1] = 10;
        }

        for (int i = 30; i < 60; i++) {
            amp[i+1] = amp[i+1-30];
            duration[i+1] = 10;
        }

        ret.put("amp", amp);
        ret.put("duration", duration);

        return ret;
    }


    // =============== Random Vibration Waveform =================
    // Android API >= 26 support pattern
    @RequiresApi(api = Build.VERSION_CODES.O)
    public void startRandomVibration() {
        long[] VIBRATE_PATTERN = new long[]{VIBRATION_DELAY*1000, VIBRATION_DURATION*1000};
        int[] VIBRATE_AMP = new int[]{0, 255};
        myVibrator.vibrate(VibrationEffect.createWaveform(
                VIBRATE_PATTERN,
                VIBRATE_AMP,
                0));
    }

    @Override
    public void onSensorChanged(SensorEvent event) {

        if (start && gravityValues != null && magneticValues != null &&
                event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {

            rawAccData[0] = event.values[0];
            rawAccData[1] = event.values[1];
            rawAccData[2] = event.values[2];

            acc_data
                .append(event.timestamp)
                .append(", ")
                .append(rawAccData[0]).append(", ").append(rawAccData[1]).append(", ").append(rawAccData[2]);

            // device coordinates to earth coordinates
            float[] earthAcc = this.transposeCoordinates(rawAccData);
            // append earth acc to data
            acc_data
                .append(", ").append(earthAcc[0]).append(", ").append(earthAcc[1]).append(", ").append(earthAcc[2])
                .append("\n");

        } else {
            if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
                gravityValues = event.values;
            }
            if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
                magneticValues = event.values;
            }
            if (!start)
                this.startButton.setBackgroundColor(Color.GRAY);
        }

        // append gyroscope data to data
        if (start && event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {

            gyro_data.append(event.timestamp)
                .append(", ")
                .append(event.values[0]).append(", ").append(event.values[1]).append(", ").append(event.values[2])
                .append("\n");
        }

        // append heart rate (using PPG sensor) data to data
        if (start && event.sensor.getType() == Sensor.TYPE_HEART_RATE) {

            heart_data.append(event.timestamp)
                .append(", ")
                .append(event.values[0])
                .append("\n");
        }

        if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
            gravityValues = event.values;
        }

        if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            magneticValues = event.values;
        }
    }


    class OutputDataThread extends Thread {
        @RequiresApi(api = Build.VERSION_CODES.O)
        @Override
        public void run() {
            super.run();

            int interval = 15000;

            while (true) {
                if (saved) {
                    start_cp();
                    saved = false;
                }

                if (start && !saved) {
                    try {
                        Thread.sleep(INTERVAL-MainActivity.this.VIBRATION_EXTENSION*1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    myVibrator.cancel();

                    try {
                        Thread.sleep(MainActivity.this.VIBRATION_EXTENSION*1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    saveToSDCard(acc_data.toString(), gyro_data.toString(), heart_data.toString());
                    times += 1;
                    start = false;
                    saved = true;

                    if (times == repeat) {
                        System.exit(0);
                    }
                }
            }
        }
    }


    private float[] transposeCoordinates(float[] deviceCoordinates) {
        float[] deviceRelativeAcc = new float[4];

        deviceRelativeAcc[0] = deviceCoordinates[0];
        deviceRelativeAcc[1] = deviceCoordinates[1];
        deviceRelativeAcc[2] = deviceCoordinates[2];
        deviceRelativeAcc[3] = 0;

        float[] R = new float[16], I = new float[16], earthAcc = new float[16];
        SensorManager.getRotationMatrix(R, I, gravityValues, magneticValues);
        float[] inv = new float[16];

        android.opengl.Matrix.invertM(inv, 0, R, 0);
        android.opengl.Matrix.multiplyMV(earthAcc, 0, inv, 0, deviceRelativeAcc, 0);

        return earthAcc;
    }


    private void lowPassFilter() {
        final float tm = System.nanoTime();
        final float dt = ((tm - this.beginTime) / 1000000000.0f) / count;
        final float alpha = rc / (rc + dt);

        if (count == 0) {
            LPFPreAccData[0] = (1 - alpha) * rawAccData[0];// x
            LPFPreAccData[1] = (1 - alpha) * rawAccData[1];// y
            LPFPreAccData[2] = (1 - alpha) * rawAccData[2];// z
        } else {
            LPFPreAccData[0] = alpha * LPFPreAccData[0] + (1 - alpha) * rawAccData[0];
            LPFPreAccData[1] = alpha * LPFPreAccData[1] + (1 - alpha) * rawAccData[1];
            LPFPreAccData[2] = alpha * LPFPreAccData[2] + (1 - alpha) * rawAccData[2];
        }

        if (start) {
            LPFAccData[0] = LPFPreAccData[0];
            LPFAccData[1] = LPFPreAccData[1];
            LPFAccData[2] = LPFPreAccData[2];
        }

        count++;
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // pass
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public void start_cp() {
        start = true;
        startButton.setBackgroundColor(Color.GREEN);
        this.beginTime = System.nanoTime();
        Date date = new Date();
        this.accOutputFileName = pre + times + "_acc.csv";
        this.gyroOutputFileName = pre + times + "_gyro.csv";
        this.heartOutputFileName = pre + times + "_heart_rate.csv";
        count = 0;

        acc_data = new StringBuilder();
        gyro_data = new StringBuilder();
        heart_data = new StringBuilder();

        // ========================= vibration ======================== //
        this.startRandomVibration();

        acc_data.append("timestamp_acc")
                .append(",").append("device_acc_x").append(",").append("device_acc_y").append(",").append("device_acc_z")
                .append(",")
                .append("earth_acc_x").append(",").append("earth_acc_y").append(",").append("earth_acc_z")
                .append("\n");

        gyro_data.append("timestamp_gyro")
                .append(",")
                .append("gyro_x").append(",").append("gyro_y").append(",").append("gyro_z")
                .append("\n");

        heart_data.append("timestamp_heart_rate")
                .append(",")
                .append("heart_rate")
                .append("\n");
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @SuppressLint({"WrongConstant", "ShowToast"})
    private void saveToSDCard(String acc_content, String gyro_content, String heart_content) {

        try {
            if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {

                ContextWrapper cw = new ContextWrapper(getApplicationContext());
                File directory = cw.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);

                File acc_file = new File(directory, accOutputFileName);
//                File gyro_file = new File(directory, gyroOutputFileName);
//                File heart_file = new File(directory, heartOutputFileName);

                System.out.println("----------> " + directory);

                String[] acc_content_stringL = acc_content.split("\n");
                if (!acc_content_stringL[0].substring(0, 13).equals("timestamp_acc")) {
                    List acc_content_list = Arrays.asList(acc_content_stringL);
                    List acc_content_headers = acc_content_list.subList(1, acc_content_list.size());
                    acc_content = String.join("\n", acc_content_headers);
                }

                String[] gyro_content_stringL = gyro_content.split("\n");
                if (!gyro_content_stringL[0].substring(0, 14).equals("timestamp_gyro")) {
                    List gyro_content_list = Arrays.asList(gyro_content_stringL);
                    List gyro_content_headers = gyro_content_list.subList(1, gyro_content_list.size());
                    gyro_content = String.join("\n", gyro_content_headers);
                }

                FileOutputStream accOutStream = new FileOutputStream(acc_file);
//                FileOutputStream gyroOutStream = new FileOutputStream(gyro_file);
//                FileOutputStream heartOutStream = new FileOutputStream(heart_file);

                accOutStream.write(acc_content.getBytes());
                accOutStream.close();

//                gyroOutStream.write(gyro_content.getBytes());
//                gyroOutStream.close();
//
//                heartOutStream.write(heart_content.getBytes());
//                heartOutStream.close();

                System.out.println(">>>>> success <<<<<");
            } else {
                System.out.println(">>>>> failed <<<<<");
            }


        } catch (Exception e) {
            System.out.println(">>>>> failed <<<<< " + e);
            e.printStackTrace();
        }
    }
}