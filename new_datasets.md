# Датасеты реальной ловкой манипуляции пятипалыми роботическими кистями

## Назначение

Список кандидатов для обучения video/world models, WAM/VLA и action decoder.
Требование: реальная роботическая кисть с пятью пальцами, временные последовательности изображений и синхронные состояния кисти на каждом timestep.

**Происхождение:** перенесено из предыдущего исследовательского ответа в переписке 29.09.2026. Это конспект для дальнейшего аудита, а не повторно проверенный каталог. Числа, лицензии, схемы и доступность ниже — сведения из того ответа; Claude должен проверить их по первичным источникам и одному эпизоду. Старые служебные ссылки вида turn... не перенесены, поскольку они не являются публичными URL.

**Пятипалая кисть ≠ измеренный угол каждого физического сустава.** Различать:
- текущие измеренные joint positions;
- целевые joint positions / команды действия;
- активные координаты механически связанных суставов;
- нормализованные driver values, которые нельзя автоматически считать углами.

Точные ссылки сохранены там, где они были в исходном ответе. Пометка «найти источник» означает, что точного URL в доступном тексте нет; имя репозитория не следует угадывать.

## 1. Основные кандидаты: богатое joint-space представление

### 1. T-Rex

- **Робот:** Dexmate Vega-1; две пятипалые Sharpa Wave, по 22 DoF.
- **Объём:** 5 464 эпизода, 5.47M кадров, около 50 часов при 30 Hz; 207 объектов; полный набор около 1.53 TB.
- **Схема:** `observation.state[58] = L_arm7 + L_hand22 + R_arm7 + R_hand22`; `action[58]` — целевые joint positions. 22 текущие координаты кисти на сторону.
- **Модальности:** 3 RGB, 10 tactile cameras, tactile deformation, 10×6D fingertip forces, language; LeRobot v3.
- **Зачем:** один из наиболее интересных источников сложной бимануальной манипуляции и высокоразмерного состояния кисти.
- **Лицензия/доступ:** в исходном обзоре указана MIT; возможен выбор отдельных эпизодов.
- **Ссылка:** https://huggingface.co/datasets/zekaiwang/trex_dataset

### 2. Dexora Real-World Dataset

- **Робот:** два AIRBOT и две XHand1, по 12 DoF; пять пальцев.
- **Объём:** 12.2K эпизодов, 2.92M кадров, 40.5 часов, около 240 GB.
- **Схема:** 39D state/action; первые 36 координат заявлены как `L_arm6 + R_arm6 + L_hand12 + R_hand12`. Проверить назначение трёх оставшихся каналов.
- **Модальности:** 4 RGB камеры: top/front/left wrist/right wrist; задачи и language.
- **Зачем:** источник бимануальных демонстраций с 12 координатами кисти на сторону.
- **Лицензия/доступ:** MIT; HF и инструменты конвертации.
- **Ссылка:** https://github.com/dexoravla/Dexora

### 3. PetalDex

- **Робот:** два 7-DoF манипулятора и две Wuji, по 20 DoF; пять пальцев.
- **Объём:** 725 эпизодов, 1.225M кадров, 30 fps, около 62 GB.
- **Схема:** `state[54] = L_arm7 + R_arm7 + L_hand20 + R_hand20`; аналогичный `action[54]`.
- **Модальности:** head RGB и wrist RGB; HDF5 и LeRobot v3.
- **Зачем:** удобная схема и высокоразмерное состояние обеих кистей.
- **Лицензия:** Apache-2.0.
- **Ссылка:** https://huggingface.co/datasets/jasonGUself/PetalDex

### 4. Robotic Origami Challenge dataset

- **Робот:** бимануальная платформа с двумя пятипалыми Sharpa Wave, по 22 DoF.
- **Объём:** около 682 эпизодов, 4.76M кадров, 30 fps.
- **Данные:** синхронные state/action кистей и манипуляторов, RGB и tactile.
- **Зачем:** длинные последовательности складывания бумаги, полезные для сложной контактной манипуляции.
- **Доступ:** в исходном обзоре указан публичный HF/LeRobot release.
- **Ссылка:** найти официальный проект и HF release по точному названию; URL и лицензию проверить.

### 5. VITRA-TeleData

- **Робот:** Realman 7-DoF и пятипалая XHand с 12 DoF.
- **Объём:** release около 5.6 GB; исходное «1K–10K samples» не уточняет, кадры это или эпизоды.
- **Схема HDF5:** `state/right_hand_joint (T, Nh)` — текущие координаты; `action/right_hand_joint (T, Nh)` — targets в radians. Предусмотрены аналогичные поля другой стороны.
- **Модальности:** RGB D455, arm qpos, EE pose, calibration, masks.
- **Зачем:** явно описанные временные состояния и целевые положения кисти.
- **Лицензия:** MIT.
- **Ссылка:** https://huggingface.co/datasets/microsoft/VITRA-TeleData

### 6. DexH2R

- **Робот:** физическая пятипалая Shadow Hand.
- **Объём:** 4 282 реальных trials, сотни тысяч кадров, 56 объектов, 39 участников.
- **Схема:** файл `qpos.pt` с Shadow Hand qpos на каждом timestep; также object pose и human reconstruction.
- **Модальности:** 18 ракурсов Kinect/RealSense/ZCam, point clouds.
- **Зачем:** динамические передачи предметов от человека роботу, движение до захвата.
- **Лицензия/доступ:** CC BY-NC 4.0; Google Drive из официального репозитория.
- **Ссылка:** https://github.com/4DVLab/DexH2R

### 7. RealDex

- **Робот:** UR10e с пятипалой Shadow Hand.
- **Объём:** около 2.6K последовательностей захвата, 955K синхронных визуальных кадров, 52 объекта.
- **Данные:** временная поза роботической кисти с joint-angle компонентой; проверить ключи, размерность и единицы в release.
- **Модальности:** 4 синхронные RGB-D камеры, point clouds, object 6D poses.
- **Зачем:** приближение к объекту и динамика закрытия кисти.
- **Доступ:** официальный downloader/GitHub/Google Drive; лицензию проверить.
- **Ссылка на статью:** https://www.alphaxiv.org/abs/2402.13853v2
- **Примечание:** это ссылка на статью; официальный dataset/download URL найти отдельно.

## 2. Крупные наборы и multi-embodiment subsets

### 8. Fourier ActionNet

- **Робот:** GR1-T1/T2, GR2 с пятипалыми Fourier hands; варианты 6 и 12 DoF на кисть.
- **Объём:** 30K+ траекторий, около 140 часов, 2.74 TB.
- **Схема HDF5:** `state/hand` и `action/hand` с 12 или 24 суммарными hand-координатами. В исходном обзоре формы записаны как `[12,x]`/`[24,x]`; ось времени проверить.
- **Модальности:** RGB H264, depth Z16, state/action и text prompt.
- **Зачем:** большой источник разнообразной реальной манипуляции; отбирать 12-DoF-hand эпизоды, если нужны более богатые состояния.
- **Лицензия/доступ:** CC BY-NC-SA 4.0; HF с принятием условий.
- **Ссылка:** https://huggingface.co/datasets/FourierIntelligence/ActionNet

### 9. AgiBot World — dexhand subset

- **Робот:** AgiBot humanoids с двумя пятипалыми dexterous hands; 6 активных координат на кисть.
- **Объём:** большой multi-embodiment набор; размер именно dexhand subset в исходном ответе не указан.
- **Схема:** `/state/effector/position` — 12 значений, left6 + right6; заявлены radians. Action также 12D.
- **Модальности:** multi-camera RGB/depth, full-body state/action, language.
- **Зачем:** масштабное обучение; обязательна фильтрация dexhand эпизодов.
- **Доступ:** HF/OpenDataLab; лицензию и точный release проверить.
- **Ссылка:** найти официальные AgiBot World dataset и schema.

### 10. RoboMIND — Tien Kung / Inspire subset

- **Робот:** Tien Kung с двумя пятипалыми Inspire.
- **Объём:** весь RoboMIND около 107K реальных траекторий, 479 задач; размер подходящего subset уточнить.
- **Схема:** `master/end_effector (T,12)`, `puppet/end_effector (T,12)`, по 6 координат на сторону; arms `joint_position (T,14)`.
- **Модальности:** RGB/depth, language, robot state/action.
- **Зачем:** большой источник реальных humanoid демонстраций.
- **Оговорка:** проверить, какие поля являются command/feedback и в каких единицах; брать Tien Kung Xsens/Inspire, исключить gripper subsets.
- **Лицензия/доступ:** Apache-2.0, gated HF.
- **Ссылка:** найти официальный RoboMIND release и Tien Kung schema.

### 11. Humanoid Everyday — H1 + Inspire

- **Робот:** только Unitree H1 с пятипалыми Inspire. G1 portion в исходном обзоре описан как трёхпалый Dex3.
- **Объём:** 260 сценариев × около 40 demos; 30 Hz; это характеристика всего набора, размер H1 subset проверить.
- **Схема:** `states.hand_state[12]`, по 6 углов на сторону; `actions.left_angles/right_angles`; есть преобразование internal representation в 12D angles.
- **Модальности:** egocentric RGB, depth, LiDAR, arm/leg state, IMU; pressure/tactile зависит от embodiment.
- **Зачем:** разнообразие бытовых задач; есть облегчённые state/action версии.
- **Ссылка:** https://github.com/physical-superintelligence-lab/Humanoid-Everyday

### 12. RoboCOIN — five_finger_hand subsets

- **Робот:** разные платформы; выбирать только embodiments с пятипалыми кистями.
- **Объём:** весь набор 180K+ реальных бимануальных demonstrations, 15 платформ. Не считать весь объём пятипалым.
- **Схема:** поля вида `left_hand_joint_*_rad`, `right_hand_joint_*_rad`; проверить фактические names/masks конкретного subset.
- **Модальности:** RGB, state/action, task metadata.
- **Доступ:** HF с access agreement; лицензия зависит от release.
- **Ссылка:** найти официальный RoboCOIN проект и коллекцию datasets.

### 13. RoboCOIN — Leju robot part placement

- **Робот:** Leju с five-finger hand.
- **Объём:** 538 эпизодов, 796 570 кадров, 30 fps, около 50 GB.
- **Данные:** RGB, state/action в общей RoboCOIN robot-joint схеме.
- **Зачем:** дополнительный subset для манипуляции деталями.
- **Лицензия/доступ:** Apache-2.0; gated HF.
- **Ссылка:** найти точный subset в RoboCOIN; проверить модель кисти, единицы и число координат.

### 14. RoboCOIN — AIRBOT MMK2 play the guitar

- **Робот:** AIRBOT MMK2 с заявленным five-finger end effector.
- **Объём:** 48 эпизодов, 8 574 кадра, 30 fps.
- **Данные:** RGB и robot joint state/action, hand fields общей схемы.
- **Зачем:** маленький источник необычной контактной задачи.
- **Ссылка:** найти точный RoboCOIN subset; проверить физическую кисть и поля состояния.

### 15. RoboCOIN — AIRBOT MMK2 storage item

- **Робот:** AIRBOT MMK2 с заявленной five-finger hand.
- **Объём:** 47 эпизодов, 8 573 кадра.
- **Данные:** RGB, hand joint state/action.
- **Зачем:** небольшой дополнительный источник манипуляции предметами.
- **Ссылка:** найти точный RoboCOIN subset; проверить embodiment и схему.

### 16. HRDexDB — inspire_f1 / inspire_dftp

- **Робот:** интересуют только пятипалые Inspire embodiments; набор содержит и другие кисти.
- **Объём:** около 2.1K grasping sequences, 100+ объектов, 5 embodiments, 23 камеры; размер Inspire subset уточнить.
- **Схема:** `raw/arm/*.npy`, `raw/hand/*.npy`, timestamps/frame IDs.
- **Модальности:** multi-camera видео, 3D annotation; tactile/contact-force для соответствующих кистей.
- **Зачем:** синхронная кинематика и многокамерные наблюдения.
- **Оговорка:** exact units, channel ordering и feedback/command convention в `raw/hand` требуют проверки.
- **Лицензия/доступ:** CC BY-NC 4.0; HF.
- **Ссылка:** https://github.com/snuvclab/HRDexDB

### 17. LET-Dex-Dataset

- **Робот:** Kuavo 4 Pro с двумя пятипалыми Linker Hand L6.
- **Объём:** около 3 000 rosbags, 20 часов, 707 GB, 14 задач.
- **Схема:** `/dexhand/state` как ROS `JointState`: 12 positions = left6 + right6; отдельно hand command.
- **Модальности:** 3 RGB-D камеры, fingertip tactile grids, wrist 6D force/torque.
- **Зачем:** реальная контактная манипуляция с тактильностью.
- **Оговорка:** 6 активных координат на руку; passive joints отдельно не измерены.
- **Доступ:** HF/AtomGit; точную ссылку и лицензию найти.

### 18. DexWild — robot portion

- **Робот:** xArm и LEAP Hand V2 Advanced; в исходном обзоре указаны 5 пальцев, 21 DoF, 17 motors.
- **Объём:** весь набор 9 505 эпизодов, 3.5M+ transitions, 33+ часов; физическая robot portion около 1 588 эпизодов / 726K transitions. Полный download порядка 2 TB.
- **Схема:** timestamped `left_leapv2.pkl`/`right_leapv2.pkl`; проверить содержимое, units и измеренный state.
- **Модальности:** ZED RGB, finger cameras, trackers; human и robot data.
- **Зачем:** human/robot mixture; для данного требования использовать физические robot episodes.
- **Лицензия/доступ:** MIT; HF.
- **Ссылка:** https://github.com/dexwild/dexwild

## 3. Дополнительные наборы с coupled hands и узкими задачами

### 19. Unitree G1 WBT Inspire — Put Clothes into Washing Machine

- **Робот:** G1 с двумя пятипалыми Inspire.
- **Объём:** 300 эпизодов, 486 315 кадров, 30 fps.
- **Схема:** `observation.state.hand_state[12]`, `action.hand_cmd[12]`; full robot current/desired q.
- **Модальности:** 4 RGB ракурса: stereo head и два wrists.
- **Зачем:** бытовая бимануальная задача с деформируемыми предметами.
- **Оговорка:** 6 active coordinates/hand; проверить единицы hand state/command.
- **Лицензия/доступ:** Apache-2.0; официальный Unitree HF release, точный URL найти.

### 20. MLeggiero / G1-GR00T Inspire pick-and-place

- **Робот:** G1 с пятипалой Inspire RH56DFTP.
- **Объём:** 632 эпизода, 331K кадров, 60 fps, 4 task/object variants.
- **Данные:** state/action с hand channels, RGB, tactile.
- **Зачем:** дополнительный реальный pick-and-place источник.
- **Ссылка:** найти HF release по MLeggiero/G1-GR00T; schema, units и лицензию проверить.

### 21. OpenArm Banana — 400 episodes

- **Робот:** OpenArm v10, две пятипалые Inspire RH56F1.
- **Объём:** 406 эпизодов, 165 349 кадров, 30 Hz, около 92 минут, 1.86 GB.
- **Схема:** 44D state/action, 6 driven finger joints на сторону; mimic/passive joints отдельно не измерены.
- **Модальности:** 4 камеры.
- **Зачем:** компактный набор для первого эксперимента.
- **Доступ:** HF/contact-gated; лицензию проверить.
- **Ссылка:** https://huggingface.co/datasets/June777/openarm_banana_400episode

### 22. G1 Pipette Tip Teleop

- **Робот:** G1 с пятипалыми Inspire.
- **Объём:** около 64 эпизодов, 57K rows.
- **Данные:** left/right hand state/action, 6 каналов на сторону, RGB/depth/tactile.
- **Оговорка:** команды заданы в hardware units 0–1000, а не radians; состояние и способ преобразования проверить отдельно.
- **Зачем:** узкая точная манипуляция, потенциально полезная для dexterity.
- **Лицензия/доступ:** Apache-2.0; публичный HF, точный URL найти.

### 23. G1 Inspire Sneaker manipulation

- **Робот:** G1 с пятипалой Inspire RH56DFTP.
- **Объём:** 94 реальных teleop эпизода, 26 192 кадра, 30 fps.
- **Данные:** 6D hand state на сторону, state/action, egocentric RGB.
- **Оговорка:** правая кисть активна, левая преимущественно открыта.
- **Лицензия:** Apache-2.0.
- **Ссылка:** https://huggingface.co/datasets/cloudwalk-research/psi0-g1-sneaker-94ep-v1

### 24. G1 Inspire Piston Pick-and-Place

- **Робот:** G1 с пятипалыми Inspire.
- **Объём:** 102 успешных эпизода, 27 143 кадра, 50 fps.
- **Схема:** state63/action30; hand/tactile channels, RGB.
- **Зачем:** небольшой предметный pick-and-place набор.
- **Оговорка:** exact channel semantics и units хуже описаны; лицензию проверить.
- **Ссылка:** https://huggingface.co/datasets/birbirll/g1-inspire-piston-pick-place

### 25. robot-dex samples — Kuavo + Linker L6

- **Робот:** Kuavo с пятипалыми Linker L6.
- **Объём:** пример release — 5 эпизодов, 3 001 кадр, 10 fps.
- **Данные:** state/action с 6 активными hand joints на сторону, head и две wrist RGB камеры.
- **Зачем:** компактный smoke test загрузки и парсинга.
- **Ссылка:** https://huggingface.co/datasets/zzzzzjhhh/robot-dex

### 26. ObjectInHand / PoseFusion

- **Робот:** физическая пятипалая Shadow Dexterous Hand.
- **Объём:** небольшой старый академический набор; точные размеры в предыдущем ответе не указаны.
- **Данные:** hand proprioception/trajectory, RGB-D, BioTac tactile, object pose.
- **Зачем:** visuo-tactile object tracking и in-hand dynamics.
- **Доступ:** Google Drive с project page; точный download, schema и лицензию найти.

## 4. Кандидаты с дополнительными вопросами к доступу или происхождению

### 27. XL-VLA dexterous dataset

- **Робот:** xArm/G1 с Ability, Inspire, XHand1, Paxini DexH13; проверить пяти-палость каждого embodiment.
- **Объём:** около 2M state-action pairs, 10 задач, 4 hand embodiments.
- **Данные:** RGB и joint-space state/action разнородных кистей.
- **Зачем:** cross-embodiment обучение.
- **Оговорка:** в предыдущем обзоре HF endpoint требовал авторизацию/вернул 401; это историческое наблюдение, текущий доступ не проверен.
- **Ссылка:** найти официальный XL-VLA проект, HF release и условия доступа.

### 28. EgoEngine / Aria-Mustard XHand subset

- **Робот:** RobotEra/XHand1, пятипалая 12-DoF кисть.
- **Объём:** десятки demonstrations, несколько тысяч кадров.
- **Данные:** 12 XHand state channels, tactile matrices, fingertip force/temp, egocentric RGB, wrist pose.
- **Зачем:** потенциальный дополнительный источник egocentric и tactile наблюдений.
- **Оговорка:** нужно подтвердить, какие последовательности физически исполнены роботом, какие получены pipeline EgoEngine; лицензия неясна.
- **Ссылка:** https://huggingface.co/datasets/Randle/EgoEngine

## 5. Все упомянутые исключённые или смежные наборы

Это также кандидаты для отдельной проверки, но исходный обзор не включал их в основной real-robot five-finger pool. Причины ниже перенесены из предыдущего ответа и не заменяют проверку конкретной версии release.

| Набор / семейство | Краткое описание и причина исключения |
| --- | --- |
| DexCap | Демонстрации ловкой манипуляции; в исходном обзоре отнесён к четырёхпалым LEAP/Allegro setups. Проверить конкретный embodiment. |
| Holo-Dex | Роботические демонстрации с Allegro Hand; четыре пальца. |
| DexWM | World-model данные с LEAP v1; четыре пальца. |
| DexFactor | Манипуляция Allegro; четыре пальца. |
| Humanoid Everyday — G1 portion | В отличие от H1/Inspire, описан как Unitree Dex3 с тремя пальцами. |
| VinT-6D | Visuo-tactile/object-pose данные; в исходном обзоре не подтверждено нужное реальное пятипалое embodiment. |
| DexGraspNet | Преимущественно синтетические/статические grasp configurations; не реальное синхронное temporal video + qpos. |
| MultiDex | Семейство grasp данных, в обзоре отнесённое к synthetic/static configurations. |
| UniDexGrasp | Синтетические grasp configurations/benchmark; не основной real-robot temporal pool. |
| BODex | Синтетические конфигурации захвата; не запись реального временного исполнения. |
| SeededGrasp | Синтетические/статические grasp данные; не real synchronized video + joints. |
| Bi-DexHands | Бимануальные dexterous задачи в simulation. |
| RP1M | Simulation; не физический real-robot dataset. |
| DexJoCo | MuJoCo; simulation. |
| VTDexManip | Human-hand trajectories; роботическая кисть в benchmark/simulation. |
| SABER-10K robot-space streams | Human video с retargeted robot configurations; qpos не является измерением физического robot execution. |
| UniDex / human-retargeted datasets | Human motion, перенесённый в robot qpos; не путать с feedback реальной роботической кисти. |
| NVIDIA Video-to-Data dexterity data | Синтетические/ретаргетированные Sharpa trajectories; полезны отдельно, но не проходят критерий real robot. |
| EgoSteer-RealWorld | Реальные две пятипалые Ruiyan RY-H2; 54 454 эпизода, около 192 часов, 20.75M кадров. В исходном обзоре hand channels описаны как normalized driver values [0,1]/[0,0.6], а не физические joint angles. Может быть полезен для video pretraining или при подтверждённой конвертации, но не засчитывать как measured joint-angle supervision. |

**EgoSteer ссылка:** https://huggingface.co/datasets/EgoSteer/EgoSteer-RealWorld

## 6. С чего начать Claude

1. Проверить основные источники T-Rex, PetalDex, Dexora, VITRA, DexH2R и RealDex: существование release, лицензию, фактические поля и один эпизод.
2. Найти недостающие точные ссылки для Origami, AgiBot, RoboMIND, RoboCOIN, LET-Dex и Unitree subsets.
3. Для каждого набора зафиксировать: robot/hand model, число пальцев, число active/measured joints, channel order, units, текущий feedback или command, FPS/timestamps и синхронизацию видео.
4. Разделить размеры полного набора и подходящего subset. Проверить overlaps/дубликаты: три RoboCOIN subsets входят в RoboCOIN; H1 subset входит в Humanoid Everyday.
5. Отдельно вести measured/native joint positions, coupled active coordinates и driver values. Не превращать шесть каналов Inspire в «20 измеренных углов».
6. Подготовить итоговый manifest с прямыми download URLs, доступными modalities, точными state/action slices, лицензией и статусом проверки.

### Приоритеты

- **Богатая кинематика кисти:** T-Rex, PetalDex, Dexora, VITRA, Origami, DexH2R, RealDex.
- **Масштаб и разнообразие:** ActionNet, AgiBot dexhand, RoboMIND Inspire, Humanoid Everyday H1, RoboCOIN five-finger, DexWild robot.
- **Компактные первые проверки:** VITRA, OpenArm Banana, robot-dex samples.
- **Вне строгого qpos pool, но полезен для видео:** EgoSteer-RealWorld.

### Предлагаемая общая запись

```text
episode_id, timestamp, images,
q_arm, q_hand,
action_arm, action_hand,
joint_names, joint_units, measured_joint_mask,
embodiment_id, hand_model, dt,
state_semantics, action_semantics,
source_dataset, source_episode, license
```

Сохранять native qpos и embodiment metadata. Общие fingertip/landmark representations можно добавить через проверенную кинематику, помечая их как вычисленные, а не измеренные.

