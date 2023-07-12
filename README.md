<style>
.rtl{
    direction: rtl;
}
.ltr{
    direction: ltr;
}
.bold{
    font-weight: bold;
}
.title1{
    font-size:30px;
}
.title2{
    font-size:22px;
}
.text{
    font-size:16px;
}
</style>

<div class="rtl bold title1">تشخیص شخص در ویدیو با استفاده از Yolov7 و Yolov8</div>
<br>
<div class="rtl bold title2">YOLO چیست؟</div>

<font size="3">
در واقع YOLO (You Only Look Once) یک مجموعه محبوب از مدل‌های تشخیص اشیاء است که برای تشخیص و طبقه‌بندی شیء در زمان واقعی که (real_time) در بینایی ماشین استفاده می‌شود.
</font>

<br>
<img src="https://howsam.org/wp-content/uploads/2019/06/object-detection.jpg"></img>
<br>

<div class="rtl text">
ویژگی کلیدی YOLO در روش تشخیص یک مرحله‌ای آن است که برای تشخیص اشیاء در زمان واقعی با دقت بالا طراحی شده است. برخلاف مدل‌های تشخیص دو مرحله‌ای مانند R-CNN که ابتدا مناطق مورد علاقه را پیشنهاد می‌دهند و سپس این مناطق را طبقه‌بندی می‌کنند، YOLO تصویر کلی را در یک عبور تحلیل می‌کند و این امر باعث افزایش سرعت و کارایی آن می‌شود.
</div>
<br>

<div class="rtl bold title2">ساختار کلی الگوریتم YOLO</div>
<div class="text">
ساختار کلی الگوریتم YOLO در شکل 5 نشان داده شده است. تصویر ورودی با ابعاد 3×448×448 به یک  Grid یا شبکه S×S تقسیم‌بندی می‌شود. این تصویر به شبکه YOLO داده می‌شود. خروجی شبکه کانولوشنی، ماتریسی به ابعاد 30×S×S خواهد بود. هریک از درایه‌های ماتریس S×S خروجی معادل با یک سلول در شبکه S×S ورودی است (به ورودی و خروجی در شکل 5 دقت کنید). خروجی 30×S×S شامل مختصات باکس‌ها و احتمال‌هاست. اگر در فرآیند آموزش (Train) باشیم، خروجی 30×S×S به‌همراه باکس‌های واقعی یا هدف (Ground Truth) به تابع اتلاف داده می‌شود. مقدار S در یولو نسخه 1، برابر با 7 درنظر گرفته شده است. اگر در فرآیند آزمایش (Test) باشیم، خروجی 30×S×S به الگوریتم حذف غیرحداکثرها (Non-maximum Suppression) داده می‌شود تا باکس‌های ضعیف از بین بروند و تنها باکس‌های درست در خروجی نمایش داده شوند.
</div>
<br>
<img src="https://howsam.org/wp-content/uploads/2019/06/2.png"></img>
<br>
<br>
<div class="rtl bold title2">شبکه YOLO</div>
<div class="text">
YOLO شامل یک شبکه عصبی کانولوشنی (Convolutional Neural Network) با 24 لایه کانولوشنی برای استخراج ویژگی و همچنین 2 لایه فولی‌کانکتد (Fully Connected) برای پیش‌بینی احتمال و مختصات اشیا است. معماری شبکه YOLO را در شکل 6 مشاهده می‌کنید.
</div>

<br>
<img src="https://howsam.org/wp-content/uploads/2019/06/1_FSFpm4f66uDq-LlEQof1Q-1024x426.png"></img>
<br>
<div class="text rtl">
همچنین، یک نسخه سریع از YOLO برای جابجایی مرزهای تشخیص اشیای سریع طراحی شده است. YOLO سریع، یک شبکه عصبی با تعداد لایه‌های کانولوشنی کمتر است که در آن از 9 لایه کانولوشنی بجای 24 لایه کانولوشنی (YOLO اصلی) استفاده شده و البته تعداد فیلترهای هر لایه در YOLO سریع نسبت به YOLO اصلی کمتر است. اندازه ورودی هر دو شبکه 3×448×448 و خروجی شبکه نیز یک تنسور 30×7×7 از پیش‌بینی‌ها است. درتمامی لایه‌ها از Leaky ReLU استفاده شده است. ممکن است سوالاتی درمورد اندازه ورودی و خروجی شبکه YOLO داشته باشید. مثلا، چرا ابعاد ورودی 3×448×448 است، درحالی‌که اکثر شبکه‌های کانولوشنی ورودی حدودا 3×224×224 دارند؟ چرا خروجی 30×7×7 است و این خروجی شامل چه اطلاعاتی است؟ چگونه از این خروجی، پیش‌بینی احتمال‌ها و مختصات باکس اشیا استخراج می‌شود؟ درادامه، در بخش آموزش شبکه YOLO به این سوالات پاسخ خواهیم داد.
</div>
<br>
<div class="rtl bold title2">مقایسه Yolov8 با مدل های قبلی</div>

<br>
<img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></img>
<br>
<br>

<div class="rtl bold title2">1.ایجاد tracker برای هر شخص با استفاده از Yolov7</div>
<br>

```linux
cd /content/drive/MyDrive/DL_projects/object-tracking/yolov7
```

```linux
pip install -r requirements.txt
```

```linux
python detect_or_track.py --weights yolov7.pt --no-trace  --source /content/drive/MyDrive/DL_projects/object-tracking/data/people.mp4 --classes 0 --track --show-track --unique-track-color --nobbox --nolabel
```

```linux
python detect_or_track.py --weights yolov7.pt --no-trace  --source /content/drive/MyDrive/DL_projects/object-tracking/data/street.mp4 --classes 0 --track --show-track --unique-track-color --nobbox --nolabel
```

```linux
python detect_or_track.py --weights yolov7.pt --no-trace  --source /content/drive/MyDrive/DL_projects/object-tracking/data/people_front.mp4 --classes 0 --track --show-track --unique-track-color --nobbox --nolabel
```

```linux
python detect_or_track.py --weights yolov7.pt --no-trace  --source /content/drive/MyDrive/DL_projects/object-tracking/data/people_top.mp4 --classes 0 --track --show-track --unique-track-color --nobbox --nolabel
```

<div class="title2 bold rtl">
2.ایجاد Bounding Box برای هر شخص به صورت دستی 
</div>
<br>

```linux
cd /content/drive/MyDrive/DL_projects/object-tracking
```

```linux
pip install ultralytics
```

```python
import cv2
import os
import random
import numpy as np
from google.colab.patches import cv2_imshow
from ultralytics import YOLO
from tracker import Tracker
```

```python
video_path1 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp4', 'people.mp4')
video_path2 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp5', 'street.mp4')
video_path3 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp6', 'people_top_view.mp4')
video_path4 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp7', 'people_front_view.mp4')

video_out1_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out1.mp4')
video_out2_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out2.mp4')
video_out3_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out3.mp4')
video_out4_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out4.mp4')

cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)
cap3 = cv2.VideoCapture(video_path3)
cap4 = cv2.VideoCapture(video_path4)

ret1 , frame1 = cap1.read()
ret2 , frame2 = cap2.read()
ret3 , frame3 = cap3.read()
ret4 , frame4 = cap4.read()

cap_out1 = cv2.VideoWriter(video_out1_path, cv2.VideoWriter_fourcc(*'MP4V'), cap1.get(cv2.CAP_PROP_FPS),(frame1.shape[1], frame1.shape[0]))
cap_out2 = cv2.VideoWriter(video_out2_path, cv2.VideoWriter_fourcc(*'MP4V'), cap2.get(cv2.CAP_PROP_FPS),(frame2.shape[1], frame2.shape[0]))
cap_out3 = cv2.VideoWriter(video_out3_path, cv2.VideoWriter_fourcc(*'MP4V'), cap3.get(cv2.CAP_PROP_FPS),(frame3.shape[1], frame3.shape[0]))
cap_out4 = cv2.VideoWriter(video_out4_path, cv2.VideoWriter_fourcc(*'MP4V'), cap4.get(cv2.CAP_PROP_FPS),(frame4.shape[1], frame4.shape[0]))
```

<div class="text rtl">
تعریف مدل
</div>
<br>

```python
model = YOLO('yolov8n.pt')
```

```python
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
detection_threshold = 0.1
```

```python
while ret1:

    results = model(frame1)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
          print(r)
          x1, y1, x2, y2, score, class_id = r
          x1 = int(x1)
          x2 = int(x2)
          y1 = int(y1)
          y2 = int(y2)
          class_id = int(class_id)
          if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame1, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cap_out1.write(frame1)
    ret1, frame1 = cap1.read()

cap1.release()
cap_out1.release()
```

```python
while ret2:

    results = model(frame2)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
          print(r)
          x1, y1, x2, y2, score, class_id = r
          x1 = int(x1)
          x2 = int(x2)
          y1 = int(y1)
          y2 = int(y2)
          class_id = int(class_id)
          if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame2, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cap_out2.write(frame2)
    ret2, frame2 = cap2.read()

cap2.release()
cap_out2.release()
```

```python
while ret3:

    results = model(frame3)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
          print(r)
          x1, y1, x2, y2, score, class_id = r
          x1 = int(x1)
          x2 = int(x2)
          y1 = int(y1)
          y2 = int(y2)
          class_id = int(class_id)
          if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame3, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame3, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cap_out3.write(frame3)
    ret3, frame3 = cap3.read()

cap3.release()
cap_out3.release()
```

```python
while ret4:

    results = model(frame4)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
          print(r)
          x1, y1, x2, y2, score, class_id = r
          x1 = int(x1)
          x2 = int(x2)
          y1 = int(y1)
          y2 = int(y2)
          class_id = int(class_id)
          if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame4, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame4, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cap_out4.write(frame4)
    ret4, frame4 = cap4.read()

cap4.release()
cap_out4.release()
```

<div class="title2 bold rtl">
3. دنبال کردن شخص با استفاده از یک id:
</div>

<div class="text rtl">
دراین مرحله ما سعی میکنیم که فقط کسانی را تشخیص دهیم که در حال حرکت هستند و یک id به هر کدام اختصاص داده می شود.
</div>
<br>

```python
video_path5 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp4', 'people.mp4')
video_path6 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp5', 'street.mp4')
video_path7 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp6', 'people_top_view.mp4')
video_path8 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp7', 'people_front_view.mp4')

cap5 = cv2.VideoCapture(video_path5)
cap6 = cv2.VideoCapture(video_path6)
cap7 = cv2.VideoCapture(video_path7)
cap8 = cv2.VideoCapture(video_path8)

ret5 , frame5 = cap5.read()
ret6 , frame6 = cap6.read()
ret7 , frame7 = cap7.read()
ret8 , frame8 = cap8.read()



video_out5_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out5.mp4')
video_out6_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out6.mp4')
video_out7_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out7.mp4')
video_out8_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out8.mp4')

cap_out5 = cv2.VideoWriter(video_out5_path, cv2.VideoWriter_fourcc(*'MP4V'), cap5.get(cv2.CAP_PROP_FPS),(frame5.shape[1], frame5.shape[0]))
cap_out6 = cv2.VideoWriter(video_out6_path, cv2.VideoWriter_fourcc(*'MP4V'), cap6.get(cv2.CAP_PROP_FPS),(frame6.shape[1], frame6.shape[0]))
cap_out7 = cv2.VideoWriter(video_out7_path, cv2.VideoWriter_fourcc(*'MP4V'), cap7.get(cv2.CAP_PROP_FPS),(frame7.shape[1], frame7.shape[0]))
cap_out8 = cv2.VideoWriter(video_out8_path, cv2.VideoWriter_fourcc(*'MP4V'), cap8.get(cv2.CAP_PROP_FPS),(frame8.shape[1], frame8.shape[0]))
```

```python
while cap5.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap5.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out5.write(annotated_frame)
    ret, frame = cap5.read()
# Release the video capture object and close the display window
cap5.release()
cap_out5.release()
```

```python
while cap6.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap6.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out6.write(annotated_frame)
    ret, frame = cap6.read()
# Release the video capture object and close the display window
cap6.release()
cap_out6.release()
```

```python
while cap7.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap7.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out7.write(annotated_frame)
    ret, frame = cap7.read()
# Release the video capture object and close the display window
cap7.release()
cap_out7.release()
```

```python
while cap8.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap8.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out8.write(annotated_frame)
    ret, frame = cap8.read()
# Release the video capture object and close the display window
cap8.release()
cap_out8.release()
```

<div class="title2 bold rtl">
4. train کردن دیتا ست دلخواه:
</div>
<div class="text rtl">
در این مرحله ما با استفاده از دیتا ست دلخواهمان مدل را train میکنیم. برای پیدا کردن دیتا ست میتوانید به سایت kaggle مراجع کنید اگر دیتا ست ما آماده نبود و میخواستیم که خودمان دیتا ست بسازیم برای این کار باید ابتدا عکس هایی از شی مورد نظرمان تهیه کنیم و سپس برای هر عکس از طریق وبسایت robflow یک محدوده مشخص کنیم و فایل خروجی را دریافت کنیم میتوانیم بخشی از دیتا را  در پوشه val قرار دهیم تا از آن برای ارزیابی مدل ترین شده استفاده کنیم.
</div>

```python
model.train(data='face_dataset_v8.yaml' , epochs=60, imgsz=640)
```

```python
video_path9 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp4', 'people.mp4')
video_path10 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp5', 'street.mp4')
video_path11 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp6', 'people_top_view.mp4')
video_path12 = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking/yolov7/runs/detect', 'exp7', 'people_front_view.mp4')

cap9 = cv2.VideoCapture(video_path9)
cap10 = cv2.VideoCapture(video_path10)
cap11 = cv2.VideoCapture(video_path11)
cap12 = cv2.VideoCapture(video_path12)

ret9 , frame9 = cap9.read()
ret10 , frame10 = cap10.read()
ret11 , frame11 = cap11.read()
ret12 , frame12 = cap12.read()



video_out9_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out9.mp4')
video_out10_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out10.mp4')
video_out11_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out11.mp4')
video_out12_path = os.path.join('/content/drive/MyDrive/DL_projects/object-tracking', 'out12.mp4')

cap_out9 = cv2.VideoWriter(video_out9_path, cv2.VideoWriter_fourcc(*'MP4V'), cap9.get(cv2.CAP_PROP_FPS),(frame9.shape[1], frame9.shape[0]))
cap_out10 = cv2.VideoWriter(video_out10_path, cv2.VideoWriter_fourcc(*'MP4V'), cap10.get(cv2.CAP_PROP_FPS),(frame10.shape[1], frame10.shape[0]))
cap_out11 = cv2.VideoWriter(video_out11_path, cv2.VideoWriter_fourcc(*'MP4V'), cap11.get(cv2.CAP_PROP_FPS),(frame11.shape[1], frame11.shape[0]))
cap_out12 = cv2.VideoWriter(video_out12_path, cv2.VideoWriter_fourcc(*'MP4V'), cap12.get(cv2.CAP_PROP_FPS),(frame12.shape[1], frame12.shape[0]))
```

```python
model = YOLO('best.pt')
```

```python
while cap9.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap9.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out9.write(annotated_frame)
    ret, frame = cap9.read()
# Release the video capture object and close the display window
cap9.release()
cap_out9.release()
```

```python
while cap10.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap10.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out10.write(annotated_frame)
    ret, frame = cap10.read()
# Release the video capture object and close the display window
cap10.release()
cap_out10.release()
```

```python
while cap11.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap11.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out11.write(annotated_frame)
    ret, frame = cap11.read()
# Release the video capture object and close the display window
cap11.release()
cap_out11.release()
```

```python
while cap12.isOpened():
# for frame in frames:
    # Read a frame from the video
    success, frame = cap12.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame , classes=[0])
        # results = model(frame5, classes = 0, )

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        # Display the annotated frame
        # cv2_imshow(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    cap_out12.write(annotated_frame)
    ret, frame = cap12.read()
# Release the video capture object and close the display window
cap12.release()
cap_out12.release()
```
