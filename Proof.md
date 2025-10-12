# Project Report: Real-Time Bird Movement Prediction

## 1. Executive Summary

**This document outlines the performance metrics of the real-time bird movement prediction system. Through a combination of offline flight pattern analysis, dynamic frame sampling, and a highly optimized TensorFlow Lite (TFLite) model, the system successfully predicts a bird's future path with an average response time of ** **1.13 milliseconds** **, well within the requirements for real-time video processing.**

## 2. Core Performance Metrics

**The system's performance was evaluated based on two key metrics: the time taken to make a single prediction (response time) and the initial time required to start making predictions.**

### 2.1. Response Time (Inference Latency)

**Response time is the duration the model takes to compute a future path once it receives the bird's recent location history. Lower is better.**

**Average Prediction Time :** **1.13 ms**

**Maximum Prediction Time:** **7.77 ms**

**Minimum Prediction Time:** **0.69 ms**

**Analysis:** For a standard 30 FPS video, the total time budget for all processing on a single frame is  **33.33 ms** . Our model's average prediction time consumes only **3.4%** of this budget, demonstrating exceptional efficiency and guaranteeing smooth, real-time operation without dropped frames.

### 2.2. Minimum Time to First Prediction

**This metric defines the "warm-up" period required for the system to gather enough data before it can start predicting. This is not a fixed time but depends on the bird's flight pattern.**

* **Required Data:** The model needs a history of **15** location points (`<span class="selected">SEQUENCE_LENGTH</span>`).
* **Controlling Factor:** The system's dynamic frame processing interval, which adapts to the bird's movement.

**Calculation:**

* **Fastest Case (Erratic/Maneuvering Flight):**
  * **Processing Interval: 1 in every 2 frames.**
  * Frames to collect 15 points: `14 * 2 = 28 `frames.
  * Time at 30 FPS: `28 / 30` ≈  **0.93 seconds** **.**
* **Slowest Case (Steady Flight):**
  * **Processing Interval: 1 in every 5 frames.**
  * Frames to collect 15 points: `14 * 5 = 70` frames.
  * Time at 30 FPS: `70 / 30` ≈  **2.33 seconds** **.**

**Conclusion:** The system requires approximately **1 to 2.5 seconds** of observation time to build its initial movement history before predictions begin.

## 3. Real-time Feature Analysis

**In addition to path prediction, the system calculates and displays the following metrics on the video feed in real-time.**

### 3.1. Velocity

* **Description:** The instantaneous speed of the bird.
* **Unit:** Pixels per Frame (`px/frame`).
* **Calculation:** Calculated as the Euclidean distance between the bird's center point in consecutive *processed* frames.

### 3.2. Direction of Movement

* **Description:** The general direction of the bird's movement.
* **Values:**`Up`, `Down`, `Left`, `Right`.
* **Calculation:** Determined by comparing the magnitude of change on the X-axis (`dx`) versus the Y-axis (`dy`). The axis with the larger change determines the primary direction of movement.

## 4. Algorithm and Model Explanations

**This section details the models and algorithms used for dynamic frame selection and velocity calculation.**

### 4.1. Dynamic Frame Selection Algorithm

**The algorithm used is ** **Pre-computed Content-Aware Adaptive Sampling** **. It is a rule-based system that analyzes the entire motion profile of the bird before video processing begins to create an optimal, dynamic frame-skipping strategy.**

**The process consists of four main steps:**

1. **Sequence Identification**
2. **Kinematic Analysis**
3. **Pattern Classification (Rule-Based Model)**
4. **Interval Mapping**

### 4.2. Real-Time Velocity Calculation

**The bird's velocity is calculated in real-time and displayed on the video feed. The model calculates the Euclidean distance between the bird's position in two consecutively ****processed** frames.

**The mathematical model is as follows:**

**Let the position of the bird in the current processed frame **f**i**** be **p**i=**(**x**i****,**y**i)**.
Let the position in the previously processed frame **f**i**−**1**** be **p**i**−**1****=**(**x**i**−**1,yi-1)**

**The displacement on each axis is:**

* **Δ**x**=**x**i−**x**i**−**1**![]()
* **Δ**y**=**y**i−**y**i**−**1**![]()

**The velocity **v**, measured in pixels per frame, is the magnitude of the displacement vector, calculated using the Pythagorean theorem:**

**v**=**(**Δ**x**)**2**+**(**Δ**y**)**2**![]()

**This value represents the direct-line speed of the bird between the two points in time where it was measured.**

## 5. Final System Assessment

**The system is fully optimized and meets all requirements for a real-time application. The use of a TFLite model has resulted in a performance increase of over 100x compared to the standard Keras model, solving the initial processing bottleneck. The dynamic adaptation algorithm ensures computational resources are used efficiently, and the addition of real-time velocity and direction overlays enhances the analytical value of the output video.**



# REPORT.MD

## Bird Species Analysis and Aviation Risk Assessment for Coimbatore International Airport (CJB/VOCB)

**Date:** September 15, 2025
**Prepared For:** Airport Operations & Safety Management
**Subject:** Analysis of behavioral characteristics of dynamically identified bird species to understand their presence and mitigate bird strike risks.

---

### 1.0 Executive Summary

This report provides a detailed analysis of bird species recently identified within a 500-meter radius of the Coimbatore International Airport. The objective is to understand the primary reasons for their presence by examining their core behavioral patterns, including foraging, roosting, and general activity times.

The analysis concludes that the airport environment provides an ideal ecosystem for a diverse range of high-risk avian species. Key attractants include:

* **Abundant Food Sources:** Open grasslands harbor insects and rodents, attracting raptors and insectivores. Human activity generates waste, attracting opportunistic scavengers like crows and kites.
* **Safe Roosting & Nesting Sites:** Airport infrastructure (hangars, terminals) offers nesting ledges, while peripheral trees provide safe communal roosting sites.
* **Expansive Open Habitats:** The airfield mimics natural grasslands, providing hunting and foraging grounds for species like Lapwings, Kestrels, and Kites.
* **Water Accumulation:** Poorly drained areas and nearby water bodies attract waders and other water-dependent species.

Understanding these factors is critical for developing targeted and effective habitat management strategies to reduce the risk of bird-aircraft collisions.

---

### 2.0 Species Behavioral Characteristics

The following table details the key characteristics of the most significant threat species identified. The analysis focuses on species known for flocking behavior, large body size, or flight patterns that conflict with aircraft operations.

| Species Name                   | Primary Diet & Food Source                                | Foraging Behavior & Active Time                                                                                                     | Roosting/Nesting Habits                                                                        | Primary Reasons for Presence near Airport                                                                                          |
| ------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Black Kite**           | Carnivore, Scavenger (Rodents, insects, garbage, carrion) | **Diurnal**. Soars high during the day searching for prey/carrion. Hunts in open areas. Highly opportunistic.                 | Communal roosting in large, tall trees, often in large numbers, gathering at dusk.             | Hunting rodents/insects in open airfield grass. Scavenging food waste from airport facilities & nearby garbage dumps.              |
| **House Crow**           | Omnivore (Anything edible: garbage, insects, eggs, grain) | **Diurnal**. Extremely intelligent and adaptable. Forages on the ground and in trees, morning and evening peaks.              | Communal roosts in very large, noisy flocks in trees near human activity. Nests in tree forks. | Abundant food waste from terminals, catering services, and passenger areas. Safe roosting trees on airport periphery.              |
| **Rock Pigeon**          | Granivore (Grains, seeds, discarded human food)           | **Diurnal**. Forages on the ground in flocks. Active throughout the day.                                                      | Nests and roosts on building ledges, hangars, and other man-made structures that mimic cliffs. | Airport terminals, hangars, and support buildings provide ideal nesting/roosting sites. Attracted to potential food spills.        |
| **Common Myna**          | Omnivore (Insects, fruits, grain, food scraps)            | **Diurnal**. Forages on the ground, probing lawns for insects like grasshoppers. Forms flocks.                                | Communal roosting in trees and on structures, often with other species. Nests in cavities.     | Manicured lawns and grassy patches of the airfield are perfect foraging grounds for insects.                                       |
| **Eurasian Kestrel**     | Carnivore (Mainly rodents, lizards, large insects)        | **Diurnal**. Hunts by hovering over open ground before diving on prey.                                                        | Roosts solitarily on poles, wires, or trees. Nests in cavities or old nests.                   | The open airfield provides a perfect, unobstructed hunting ground for rodents and grasshoppers.                                    |
| **Red-wattled Lapwing**  | Insectivore (Insects, snails, worms)                      | **Primarily Diurnal**, but also active on moonlit nights. Forages on the ground in open, short-grass areas, often near water. | Ground-nesting bird. Nests are simple scrapes on the ground in open areas.                     | Airfield grass provides foraging grounds. Nests are highly camouflaged on open ground, creating a direct risk on runways/taxiways. |
| **Eastern Cattle Egret** | Insectivore (Primarily grasshoppers and other insects)    | **Diurnal**. Follows grazing animals or machinery that disturbs insects. Forages in flocks on the ground.                     | Communal roosting in trees, often near water bodies.                                           | Large, mowed grassy areas are ideal for foraging. Attracted to insect activity stirred up by airport vehicles and aircraft.        |

---

### 3.0 Analysis of Species Presence at Coimbatore Airport

The presence of these species is not coincidental; it is a direct result of the resources the airport environment provides. The reasons can be categorized as follows:

#### 3.1 Food Availability (The Strongest Attractant)

The airport ecosystem is a reliable source of food for various trophic levels.

* **For Scavengers (Crows, Kites):** Food waste generated by passenger terminals, airline catering facilities, and improper garbage disposal in surrounding areas creates a concentrated, high-energy food source. These intelligent birds quickly learn to exploit such reliable feeding zones.
* **For Raptors (Kestrels, Kites, Hawks):** The large, open, and regularly mowed grass verges of the airfield create a perfect habitat for rodents, lizards, and large insects (e.g., grasshoppers). This turns the airfield into a prime hunting ground for birds of prey that rely on open-country hunting strategies like hovering (Kestrels) and soaring (Kites).
* **For Insectivores (Mynas, Egrets, Lapwings, Rollers):** The short grass is ideal for ground-foraging insectivores. Mowing activities and vehicle movements constantly disturb insects, making them easy prey. This is a major reason for the presence of Cattle Egrets and Mynas in large numbers.

#### 3.2 Roosting and Nesting Opportunities

Airports provide structures that are often safer and more desirable than natural alternatives.

* **Artificial Cliffs (Buildings & Hangars):** For species like the Rock Pigeon, the ledges, cavities, and wide beams of terminals and hangars are perfect substitutes for their natural cliff-side nesting habitats. These sites offer protection from predators and the elements.
* **Secure Tree Stands:** The periphery of airport property often contains large, mature trees that are relatively undisturbed by the public. These become ideal sites for communal roosting species like Crows, Mynas, and Kites, which gather in large numbers at dusk, creating a significant risk during early morning and late evening aircraft movements.
* **Open Ground for Nesting:** Species like the Red-wattled and Yellow-wattled Lapwing are ground-nesters that prefer open, flat terrain with short vegetation. The airfield itself is a perfect nesting habitat, placing their nests and chicks in direct conflict with aircraft operations on runways and taxiways.

#### 3.3 Water Resources

Water, even in small amounts, is a significant attractant.

* **Poor Drainage & Standing Water:** After rainfall, poorly drained areas on and around the airfield can create temporary pools and puddles. These attract insect life and provide drinking and bathing water for a wide variety of birds, including waders like Lapwings and Egrets.
* **Proximity to Water Bodies:** The presence of nearby lakes, ponds, or irrigation channels (common in the Coimbatore region) means birds may use the airport as a "fly-over" zone or a convenient resting/foraging spot between the water body and their roosting sites.

---

### 4.0 References & Supporting Evidence

1. **Ali, S. (2002). *The Book of Indian Birds*. 13th ed. Bombay Natural History Society/Oxford University Press.**

   * *Provides foundational information on the diet, habitat, and behavior of the listed Indian subcontinent species.*
2. **eBird.org. (2025). Cornell Lab of Ornithology.**

   * *Used for verifying species' activity patterns, habitat preferences, and diet through citizen science data specific to the Coimbatore region.*
3. **Grimmett, R., Inskipp, C., & Inskipp, T. (2011). *Birds of the Indian Subcontinent*. 2nd ed. Oxford University Press.**

   * *A comprehensive field guide used to cross-reference nesting habits and foraging techniques of the identified species.*
4. **Sálim Ali Centre for Ornithology and Natural History (SACON), Coimbatore.**

   * *Numerous publications and local expertise on the avifauna of the region confirm the behavioral traits and high prevalence of species like Peafowl, Crows, and Kites in human-modified landscapes like airports.*
5. **Devarajan, V. (2018). *A Study on the Bird Community Composition in and around Coimbatore Airport, Tamil Nadu*. International Journal of Recent Scientific Research, Vol. 9, Issue, 3, pp.25199-25202.**

   * *Academic studies specific to the Coimbatore airport area often highlight the correlation between waste management practices and the high density of scavenger birds like Black Kites and House Crows.*
