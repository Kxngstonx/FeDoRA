# Experiment Results Summary

## Dataset: anli

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| anli_FFA-LoRA | 10 | 0.3330 | 0.3339 | 0.3332 | 0.3330 | 0.3330 | 0.3330 |
| anli_FFA-LoRA | 20 | 0.3890 | 0.3440 | 0.3603 | 0.3725 | 0.3812 | 0.3870 |
| anli_FFA-LoRA | 30 | 0.4520 | 0.4070 | 0.4449 | 0.4506 | 0.4518 | 0.4520 |
| anli_FL+DoRA(FlexLoRA) | 10 | 0.4670 | 0.3541 | 0.3905 | 0.4206 | 0.4442 | 0.4611 |
| anli_FL+DoRA(FlexLoRA) | 20 | 0.5670 | 0.4821 | 0.5566 | 0.5670 | 0.5686 | 0.5678 |
| anli_FL+DoRA(FlexLoRA) | 30 | 0.5840 | 0.5464 | 0.5821 | 0.5836 | 0.5838 | 0.5840 |
| anli_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.3330 | 0.3335 | 0.3331 | 0.3330 | 0.3330 | 0.3330 |
| anli_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.4460 | 0.3666 | 0.4090 | 0.4309 | 0.4411 | 0.4452 |
| anli_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.4780 | 0.4332 | 0.4723 | 0.4769 | 0.4779 | 0.4780 |
| anli_FeDoRA | 10 | 0.4520 | 0.3455 | 0.3689 | 0.3925 | 0.4163 | 0.4401 |
| anli_FeDoRA | 20 | 0.5530 | 0.4670 | 0.5396 | 0.5512 | 0.5537 | 0.5536 |
| anli_FeDoRA | 30 | 0.5690 | 0.5292 | 0.5658 | 0.5687 | 0.5693 | 0.5691 |
| anli_FedEx-LoRA | 10 | 0.3330 | 0.3339 | 0.3332 | 0.3330 | 0.3330 | 0.3330 |
| anli_FedEx-LoRA | 20 | 0.5210 | 0.4251 | 0.4980 | 0.5156 | 0.5196 | 0.5207 |
| anli_FedEx-LoRA | 30 | 0.5500 | 0.4988 | 0.5424 | 0.5476 | 0.5495 | 0.5500 |
| anli_FedIT | 10 | 0.3330 | 0.3340 | 0.3332 | 0.3330 | 0.3330 | 0.3330 |
| anli_FedIT | 20 | 0.5370 | 0.4397 | 0.5134 | 0.5302 | 0.5352 | 0.5368 |
| anli_FedIT | 30 | 0.5610 | 0.5152 | 0.5580 | 0.5609 | 0.5613 | 0.5611 |
| anli_FedIT | 40 | 0.3330 | 0.3968 | 0.3395 | 0.3333 | 0.3330 | 0.3330 |
| anli_FedIT | 50 | 0.5370 | 0.4617 | 0.5136 | 0.5302 | 0.5352 | 0.5368 |
| anli_FedIT | 60 | 0.5610 | 0.5229 | 0.5580 | 0.5609 | 0.5613 | 0.5611 |
| anli_FlexLoRA | 10 | 0.3700 | 0.3374 | 0.3442 | 0.3515 | 0.3589 | 0.3663 |
| anli_FlexLoRA | 20 | 0.5430 | 0.4530 | 0.5249 | 0.5381 | 0.5415 | 0.5428 |
| anli_FlexLoRA | 30 | 0.5540 | 0.5152 | 0.5501 | 0.5526 | 0.5536 | 0.5540 |

## Dataset: glue_mnli

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| glue_mnli_FL+DoRA(FlexLoRA) | 10 | 0.8043 | 0.5075 | 0.6911 | 0.7685 | 0.7949 | 0.8025 |
| glue_mnli_FL+DoRA(FlexLoRA) | 20 | 0.8470 | 0.7226 | 0.8381 | 0.8450 | 0.8463 | 0.8469 |
| glue_mnli_FL+DoRA(FlexLoRA) | 30 | 0.8508 | 0.8055 | 0.8500 | 0.8507 | 0.8508 | 0.8508 |
| glue_mnli_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.7292 | 0.4324 | 0.5635 | 0.6492 | 0.6973 | 0.7217 |
| glue_mnli_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.8068 | 0.6651 | 0.7915 | 0.8029 | 0.8053 | 0.8064 |
| glue_mnli_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.8086 | 0.7598 | 0.8092 | 0.8090 | 0.8087 | 0.8086 |
| glue_mnli_FeDoRA | 10 | 0.8025 | 0.5014 | 0.6814 | 0.7616 | 0.7909 | 0.8002 |
| glue_mnli_FeDoRA | 20 | 0.8470 | 0.7202 | 0.8377 | 0.8449 | 0.8462 | 0.8468 |
| glue_mnli_FeDoRA | 30 | 0.8499 | 0.8044 | 0.8494 | 0.8499 | 0.8500 | 0.8499 |
| glue_mnli_FedIT | 10 | 0.7779 | 0.4526 | 0.6094 | 0.7070 | 0.7563 | 0.7749 |
| glue_mnli_FedIT | 20 | 0.8409 | 0.6981 | 0.8294 | 0.8390 | 0.8404 | 0.8408 |
| glue_mnli_FedIT | 30 | 0.8445 | 0.7938 | 0.8444 | 0.8446 | 0.8445 | 0.8445 |
| glue_mnli_FedIT | 40 | 0.7779 | 0.6049 | 0.6232 | 0.7074 | 0.7563 | 0.7749 |
| glue_mnli_FedIT | 50 | 0.8409 | 0.7512 | 0.8298 | 0.8390 | 0.8404 | 0.8408 |
| glue_mnli_FedIT | 60 | 0.8445 | 0.8123 | 0.8444 | 0.8446 | 0.8445 | 0.8445 |
| glue_mnli_FFA-LoRA | 10 | 0.6907 | 0.4161 | 0.5253 | 0.6028 | 0.6511 | 0.6802 |
| glue_mnli_FFA-LoRA | 20 | 0.8004 | 0.6533 | 0.7829 | 0.7958 | 0.7986 | 0.7999 |
| glue_mnli_FFA-LoRA | 30 | 0.8024 | 0.7515 | 0.8028 | 0.8028 | 0.8026 | 0.8024 |
| glue_mnli_FedEx-LoRA | 10 | 0.7934 | 0.4715 | 0.6417 | 0.7356 | 0.7773 | 0.7912 |
| glue_mnli_FedEx-LoRA | 20 | 0.8427 | 0.7072 | 0.8332 | 0.8414 | 0.8424 | 0.8427 |
| glue_mnli_FedEx-LoRA | 30 | 0.8487 | 0.7991 | 0.8481 | 0.8486 | 0.8487 | 0.8487 |
| glue_mnli_FlexLoRA | 10 | 0.7080 | 0.4162 | 0.5308 | 0.6152 | 0.6687 | 0.6989 |
| glue_mnli_FlexLoRA | 20 | 0.8244 | 0.6680 | 0.8061 | 0.8199 | 0.8227 | 0.8240 |
| glue_mnli_FlexLoRA | 30 | 0.8296 | 0.7736 | 0.8296 | 0.8301 | 0.8299 | 0.8297 |

## Dataset: glue_qqp

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| glue_qqp_FL+DoRA(FlexLoRA) | 10 | 0.8241 | 0.7184 | 0.7922 | 0.8133 | 0.8201 | 0.8232 |
| glue_qqp_FL+DoRA(FlexLoRA) | 20 | 0.8426 | 0.7948 | 0.8378 | 0.8411 | 0.8421 | 0.8426 |
| glue_qqp_FL+DoRA(FlexLoRA) | 30 | 0.8451 | 0.8279 | 0.8453 | 0.8454 | 0.8453 | 0.8451 |
| glue_qqp_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.7818 | 0.6956 | 0.7532 | 0.7715 | 0.7779 | 0.7808 |
| glue_qqp_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.8092 | 0.7657 | 0.8070 | 0.8108 | 0.8108 | 0.8099 |
| glue_qqp_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.8060 | 0.7959 | 0.8095 | 0.8077 | 0.8066 | 0.8061 |
| glue_qqp_FeDoRA | 10 | 0.8098 | 0.7037 | 0.7732 | 0.7972 | 0.8056 | 0.8089 |
| glue_qqp_FeDoRA | 20 | 0.8396 | 0.7846 | 0.8316 | 0.8366 | 0.8382 | 0.8392 |
| glue_qqp_FeDoRA | 30 | 0.8376 | 0.8205 | 0.8390 | 0.8385 | 0.8379 | 0.8377 |
| glue_qqp_FedIT | 10 | 0.8136 | 0.7045 | 0.7757 | 0.8004 | 0.8093 | 0.8129 |
| glue_qqp_FedIT | 20 | 0.8410 | 0.7878 | 0.8340 | 0.8379 | 0.8394 | 0.8405 |
| glue_qqp_FedIT | 30 | 0.8436 | 0.8242 | 0.8436 | 0.8439 | 0.8437 | 0.8436 |
| glue_qqp_FedIT | 40 | 0.8136 | 0.7716 | 0.7817 | 0.8006 | 0.8093 | 0.8129 |
| glue_qqp_FedIT | 50 | 0.8410 | 0.8112 | 0.8342 | 0.8379 | 0.8394 | 0.8405 |
| glue_qqp_FedIT | 60 | 0.8436 | 0.8324 | 0.8436 | 0.8439 | 0.8437 | 0.8436 |
| glue_qqp_FFA-LoRA | 10 | 0.7699 | 0.6809 | 0.7356 | 0.7583 | 0.7667 | 0.7694 |
| glue_qqp_FFA-LoRA | 20 | 0.8108 | 0.7602 | 0.8061 | 0.8110 | 0.8118 | 0.8113 |
| glue_qqp_FFA-LoRA | 30 | 0.7986 | 0.7916 | 0.8041 | 0.8008 | 0.7992 | 0.7987 |
| glue_qqp_FedEx-LoRA | 10 | 0.8147 | 0.7066 | 0.7783 | 0.8029 | 0.8114 | 0.8142 |
| glue_qqp_FedEx-LoRA | 20 | 0.8434 | 0.7902 | 0.8377 | 0.8419 | 0.8429 | 0.8433 |
| glue_qqp_FedEx-LoRA | 30 | 0.8400 | 0.8248 | 0.8421 | 0.8412 | 0.8404 | 0.8401 |
| glue_qqp_FlexLoRA | 10 | 0.8043 | 0.7071 | 0.7754 | 0.7963 | 0.8024 | 0.8041 |
| glue_qqp_FlexLoRA | 20 | 0.8364 | 0.7857 | 0.8301 | 0.8341 | 0.8354 | 0.8361 |
| glue_qqp_FlexLoRA | 30 | 0.8333 | 0.8184 | 0.8348 | 0.8341 | 0.8335 | 0.8333 |

## Dataset: glue_sst2

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| glue_sst2_FL+DoRA(FlexLoRA) | 10 | 0.9323 | 0.7611 | 0.9149 | 0.9343 | 0.9344 | 0.9330 |
| glue_sst2_FL+DoRA(FlexLoRA) | 20 | 0.9404 | 0.8780 | 0.9400 | 0.9411 | 0.9411 | 0.9407 |
| glue_sst2_FL+DoRA(FlexLoRA) | 30 | 0.9415 | 0.9203 | 0.9422 | 0.9418 | 0.9416 | 0.9415 |
| glue_sst2_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.9312 | 0.7439 | 0.9014 | 0.9265 | 0.9294 | 0.9305 |
| glue_sst2_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.9427 | 0.8731 | 0.9421 | 0.9437 | 0.9435 | 0.9429 |
| glue_sst2_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.9461 | 0.9210 | 0.9463 | 0.9462 | 0.9461 | 0.9461 |
| glue_sst2_FeDoRA | 10 | 0.9392 | 0.7589 | 0.9159 | 0.9383 | 0.9401 | 0.9396 |
| glue_sst2_FeDoRA | 20 | 0.9472 | 0.8805 | 0.9456 | 0.9469 | 0.9470 | 0.9471 |
| glue_sst2_FeDoRA | 30 | 0.9404 | 0.9206 | 0.9413 | 0.9407 | 0.9404 | 0.9404 |
| glue_sst2_FedIT | 10 | 0.9461 | 0.7518 | 0.9152 | 0.9417 | 0.9451 | 0.9458 |
| glue_sst2_FedIT | 20 | 0.9450 | 0.8777 | 0.9443 | 0.9451 | 0.9451 | 0.9451 |
| glue_sst2_FedIT | 30 | 0.9472 | 0.9228 | 0.9469 | 0.9471 | 0.9472 | 0.9472 |
| glue_sst2_FedIT | 40 | 0.9461 | 0.8960 | 0.9275 | 0.9422 | 0.9451 | 0.9458 |
| glue_sst2_FedIT | 50 | 0.9450 | 0.9280 | 0.9447 | 0.9451 | 0.9451 | 0.9451 |
| glue_sst2_FedIT | 60 | 0.9472 | 0.9403 | 0.9469 | 0.9471 | 0.9472 | 0.9472 |
| glue_sst2_FFA-LoRA | 10 | 0.9404 | 0.7438 | 0.9056 | 0.9338 | 0.9383 | 0.9399 |
| glue_sst2_FFA-LoRA | 20 | 0.9450 | 0.8734 | 0.9424 | 0.9440 | 0.9445 | 0.9449 |
| glue_sst2_FFA-LoRA | 30 | 0.9461 | 0.9214 | 0.9464 | 0.9462 | 0.9461 | 0.9461 |
| glue_sst2_FedEx-LoRA | 10 | 0.9472 | 0.7507 | 0.9140 | 0.9411 | 0.9451 | 0.9466 |
| glue_sst2_FedEx-LoRA | 20 | 0.9415 | 0.8777 | 0.9443 | 0.9443 | 0.9433 | 0.9422 |
| glue_sst2_FedEx-LoRA | 30 | 0.9472 | 0.9229 | 0.9470 | 0.9471 | 0.9472 | 0.9472 |
| glue_sst2_FlexLoRA | 10 | 0.9415 | 0.7539 | 0.9130 | 0.9374 | 0.9404 | 0.9412 |
| glue_sst2_FlexLoRA | 20 | 0.9427 | 0.8775 | 0.9436 | 0.9446 | 0.9441 | 0.9432 |
| glue_sst2_FlexLoRA | 30 | 0.9472 | 0.9226 | 0.9469 | 0.9471 | 0.9472 | 0.9472 |

## Dataset: mmlu

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| mmlu_FFA-LoRA | 10 | 0.2554 | 0.2545 | 0.2553 | 0.2555 | 0.2555 | 0.2555 |
| mmlu_FFA-LoRA | 20 | 0.2549 | 0.2548 | 0.2549 | 0.2548 | 0.2548 | 0.2549 |
| mmlu_FFA-LoRA | 30 | 0.2584 | 0.2562 | 0.2577 | 0.2582 | 0.2584 | 0.2584 |
| mmlu_FeDoRA | 10 | 0.2551 | 0.2528 | 0.2545 | 0.2546 | 0.2547 | 0.2549 |
| mmlu_FeDoRA | 20 | 0.2497 | 0.2534 | 0.2531 | 0.2522 | 0.2512 | 0.2502 |
| mmlu_FeDoRA | 30 | 0.2483 | 0.2516 | 0.2494 | 0.2485 | 0.2483 | 0.2483 |
| mmlu_FedEx-LoRA | 10 | 0.2551 | 0.2557 | 0.2562 | 0.2564 | 0.2562 | 0.2556 |
| mmlu_FedEx-LoRA | 20 | 0.2523 | 0.2553 | 0.2546 | 0.2540 | 0.2534 | 0.2527 |
| mmlu_FedEx-LoRA | 30 | 0.2517 | 0.2541 | 0.2524 | 0.2517 | 0.2516 | 0.2516 |
| mmlu_FedIT | 10 | 0.2551 | 0.2553 | 0.2554 | 0.2554 | 0.2553 | 0.2552 |
| mmlu_FedIT | 20 | 0.2485 | 0.2542 | 0.2524 | 0.2509 | 0.2497 | 0.2488 |
| mmlu_FedIT | 30 | 0.2550 | 0.2551 | 0.2553 | 0.2552 | 0.2551 | 0.2550 |
| mmlu_FedIT | 40 | 0.2551 | 0.2553 | 0.2554 | 0.2554 | 0.2553 | 0.2552 |
| mmlu_FedIT | 50 | 0.2485 | 0.2542 | 0.2524 | 0.2509 | 0.2497 | 0.2488 |
| mmlu_FedIT | 60 | 0.2550 | 0.2551 | 0.2553 | 0.2552 | 0.2551 | 0.2550 |
| mmlu_FlexLoRA | 10 | 0.2551 | 0.2555 | 0.2557 | 0.2558 | 0.2557 | 0.2553 |
| mmlu_FlexLoRA | 20 | 0.2534 | 0.2544 | 0.2536 | 0.2533 | 0.2532 | 0.2533 |
| mmlu_FlexLoRA | 30 | 0.2491 | 0.2530 | 0.2508 | 0.2497 | 0.2492 | 0.2491 |

## Dataset: superglue_boolq

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| superglue_boolq_FFA-LoRA | 10 | 0.6835 | 0.6333 | 0.6525 | 0.6667 | 0.6764 | 0.6821 |
| superglue_boolq_FFA-LoRA | 20 | 0.7740 | 0.7137 | 0.7624 | 0.7703 | 0.7725 | 0.7735 |
| superglue_boolq_FFA-LoRA | 30 | 0.7725 | 0.7537 | 0.7738 | 0.7733 | 0.7729 | 0.7726 |
| superglue_boolq_FFA-LoRA_seed43 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FFA-LoRA_seed43 | 20 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FFA-LoRA_seed43 | 30 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FFA-LoRA_seed44 | 10 | 0.6217 | 0.6217 | 0.6218 | 0.6218 | 0.6218 | 0.6217 |
| superglue_boolq_FFA-LoRA_seed44 | 20 | 0.6486 | 0.6320 | 0.6429 | 0.6473 | 0.6488 | 0.6489 |
| superglue_boolq_FFA-LoRA_seed44 | 30 | 0.6508 | 0.6488 | 0.6545 | 0.6524 | 0.6512 | 0.6508 |
| superglue_boolq_FL+DoRA(FlexLoRA) | 10 | 0.7823 | 0.6707 | 0.7330 | 0.7634 | 0.7763 | 0.7811 |
| superglue_boolq_FL+DoRA(FlexLoRA) | 20 | 0.8095 | 0.7574 | 0.8048 | 0.8085 | 0.8092 | 0.8094 |
| superglue_boolq_FL+DoRA(FlexLoRA) | 30 | 0.8138 | 0.7941 | 0.8137 | 0.8140 | 0.8139 | 0.8138 |
| superglue_boolq_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.6810 | 0.6326 | 0.6507 | 0.6644 | 0.6739 | 0.6795 |
| superglue_boolq_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.7725 | 0.7137 | 0.7618 | 0.7691 | 0.7711 | 0.7721 |
| superglue_boolq_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.7807 | 0.7566 | 0.7791 | 0.7799 | 0.7803 | 0.7806 |
| superglue_boolq_FeDoRA | 10 | 0.7771 | 0.6670 | 0.7261 | 0.7564 | 0.7701 | 0.7757 |
| superglue_boolq_FeDoRA | 20 | 0.8028 | 0.7521 | 0.7986 | 0.8021 | 0.8025 | 0.8027 |
| superglue_boolq_FeDoRA | 30 | 0.8064 | 0.7883 | 0.8071 | 0.8069 | 0.8066 | 0.8064 |
| superglue_boolq_FeDoRA_seed43 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FeDoRA_seed43 | 20 | 0.6220 | 0.6218 | 0.6219 | 0.6219 | 0.6220 | 0.6220 |
| superglue_boolq_FeDoRA_seed43 | 30 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FeDoRA_seed44 | 10 | 0.6217 | 0.6179 | 0.6198 | 0.6214 | 0.6217 | 0.6217 |
| superglue_boolq_FeDoRA_seed44 | 20 | 0.6927 | 0.6445 | 0.6738 | 0.6865 | 0.6915 | 0.6928 |
| superglue_boolq_FeDoRA_seed44 | 30 | 0.7177 | 0.6909 | 0.7169 | 0.7186 | 0.7183 | 0.7178 |
| superglue_boolq_FedEx-LoRA | 10 | 0.7823 | 0.6682 | 0.7294 | 0.7610 | 0.7753 | 0.7810 |
| superglue_boolq_FedEx-LoRA | 20 | 0.8110 | 0.7567 | 0.8050 | 0.8090 | 0.8100 | 0.8107 |
| superglue_boolq_FedEx-LoRA | 30 | 0.8131 | 0.7939 | 0.8138 | 0.8139 | 0.8135 | 0.8132 |
| superglue_boolq_FedEx-LoRA_seed43 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedEx-LoRA_seed43 | 20 | 0.6220 | 0.6217 | 0.6218 | 0.6219 | 0.6219 | 0.6220 |
| superglue_boolq_FedEx-LoRA_seed43 | 30 | 0.6217 | 0.6214 | 0.6215 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedEx-LoRA_seed44 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedEx-LoRA_seed44 | 20 | 0.7052 | 0.6524 | 0.6848 | 0.6980 | 0.7033 | 0.7050 |
| superglue_boolq_FedEx-LoRA_seed44 | 30 | 0.7352 | 0.7032 | 0.7324 | 0.7351 | 0.7353 | 0.7352 |
| superglue_boolq_FedIT | 10 | 0.7786 | 0.6665 | 0.7259 | 0.7568 | 0.7711 | 0.7771 |
| superglue_boolq_FedIT | 20 | 0.8092 | 0.7554 | 0.8040 | 0.8080 | 0.8087 | 0.8090 |
| superglue_boolq_FedIT | 30 | 0.8156 | 0.7931 | 0.8140 | 0.8151 | 0.8155 | 0.8156 |
| superglue_boolq_FedIT | 40 | 0.7786 | 0.7263 | 0.7313 | 0.7570 | 0.7711 | 0.7771 |
| superglue_boolq_FedIT | 50 | 0.8092 | 0.7762 | 0.8041 | 0.8080 | 0.8087 | 0.8090 |
| superglue_boolq_FedIT | 60 | 0.8156 | 0.8003 | 0.8140 | 0.8151 | 0.8155 | 0.8156 |
| superglue_boolq_FedIT_seed43 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed43 | 20 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed43 | 30 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed43 | 40 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed43 | 50 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed43 | 60 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed44 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed44 | 20 | 0.7086 | 0.6522 | 0.6871 | 0.7019 | 0.7073 | 0.7086 |
| superglue_boolq_FedIT_seed44 | 30 | 0.7318 | 0.7015 | 0.7286 | 0.7309 | 0.7314 | 0.7317 |
| superglue_boolq_FedIT_seed44 | 40 | 0.6217 | 0.6496 | 0.6247 | 0.6218 | 0.6217 | 0.6217 |
| superglue_boolq_FedIT_seed44 | 50 | 0.7086 | 0.6619 | 0.6872 | 0.7019 | 0.7073 | 0.7086 |
| superglue_boolq_FedIT_seed44 | 60 | 0.7318 | 0.7048 | 0.7286 | 0.7309 | 0.7314 | 0.7317 |
| superglue_boolq_FlexLoRA | 10 | 0.7624 | 0.6577 | 0.7090 | 0.7390 | 0.7546 | 0.7612 |
| superglue_boolq_FlexLoRA | 20 | 0.8040 | 0.7477 | 0.7977 | 0.8026 | 0.8036 | 0.8039 |
| superglue_boolq_FlexLoRA | 30 | 0.8058 | 0.7867 | 0.8067 | 0.8063 | 0.8059 | 0.8058 |
| superglue_boolq_FlexLoRA_seed43 | 10 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FlexLoRA_seed43 | 20 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FlexLoRA_seed43 | 30 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 | 0.6217 |
| superglue_boolq_FlexLoRA_seed44 | 10 | 0.6217 | 0.6217 | 0.6216 | 0.6216 | 0.6216 | 0.6217 |
| superglue_boolq_FlexLoRA_seed44 | 20 | 0.7190 | 0.6610 | 0.6992 | 0.7126 | 0.7173 | 0.7188 |
| superglue_boolq_FlexLoRA_seed44 | 30 | 0.7422 | 0.7121 | 0.7405 | 0.7425 | 0.7424 | 0.7423 |

## Dataset: superglue_multirc

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| superglue_multirc_FFA-LoRA | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FFA-LoRA | 20 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FFA-LoRA | 30 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FFA-LoRA_seed43 | 10 | 0.5720 | 0.5644 | 0.5655 | 0.5700 | 0.5718 | 0.5720 |
| superglue_multirc_FFA-LoRA_seed43 | 20 | 0.5720 | 0.5693 | 0.5718 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FFA-LoRA_seed43 | 30 | 0.5720 | 0.5711 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FFA-LoRA_seed44 | 10 | 0.5571 | 0.4870 | 0.5228 | 0.5367 | 0.5454 | 0.5528 |
| superglue_multirc_FFA-LoRA_seed44 | 20 | 0.5307 | 0.5278 | 0.5483 | 0.5445 | 0.5389 | 0.5334 |
| superglue_multirc_FFA-LoRA_seed44 | 30 | 0.5637 | 0.5460 | 0.5595 | 0.5624 | 0.5635 | 0.5637 |
| superglue_multirc_FL+DoRA(FlexLoRA) | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FL+DoRA(FlexLoRA) | 20 | 0.5720 | 0.5713 | 0.5729 | 0.5732 | 0.5729 | 0.5723 |
| superglue_multirc_FL+DoRA(FlexLoRA) | 30 | 0.5726 | 0.5720 | 0.5724 | 0.5724 | 0.5725 | 0.5726 |
| superglue_multirc_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FeDoRA | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FeDoRA | 20 | 0.5720 | 0.5718 | 0.5716 | 0.5716 | 0.5716 | 0.5718 |
| superglue_multirc_FeDoRA | 30 | 0.5681 | 0.5708 | 0.5694 | 0.5686 | 0.5682 | 0.5681 |
| superglue_multirc_FeDoRA_seed43 | 10 | 0.5720 | 0.5672 | 0.5701 | 0.5717 | 0.5719 | 0.5719 |
| superglue_multirc_FeDoRA_seed43 | 20 | 0.5639 | 0.5633 | 0.5630 | 0.5632 | 0.5637 | 0.5640 |
| superglue_multirc_FeDoRA_seed43 | 30 | 0.5553 | 0.5603 | 0.5569 | 0.5558 | 0.5555 | 0.5554 |
| superglue_multirc_FeDoRA_seed44 | 10 | 0.5710 | 0.4800 | 0.5128 | 0.5319 | 0.5476 | 0.5624 |
| superglue_multirc_FeDoRA_seed44 | 20 | 0.5157 | 0.5175 | 0.5356 | 0.5308 | 0.5250 | 0.5191 |
| superglue_multirc_FeDoRA_seed44 | 30 | 0.5602 | 0.5385 | 0.5540 | 0.5575 | 0.5591 | 0.5599 |
| superglue_multirc_FedEx-LoRA | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FedEx-LoRA | 20 | 0.5726 | 0.5712 | 0.5705 | 0.5705 | 0.5710 | 0.5719 |
| superglue_multirc_FedEx-LoRA | 30 | 0.5736 | 0.5726 | 0.5735 | 0.5736 | 0.5736 | 0.5736 |
| superglue_multirc_FedEx-LoRA_seed43 | 10 | 0.5720 | 0.5670 | 0.5694 | 0.5712 | 0.5715 | 0.5718 |
| superglue_multirc_FedEx-LoRA_seed43 | 20 | 0.5592 | 0.5692 | 0.5685 | 0.5660 | 0.5633 | 0.5605 |
| superglue_multirc_FedEx-LoRA_seed43 | 30 | 0.5722 | 0.5712 | 0.5722 | 0.5724 | 0.5723 | 0.5722 |
| superglue_multirc_FedEx-LoRA_seed44 | 10 | 0.5699 | 0.4936 | 0.5402 | 0.5583 | 0.5668 | 0.5696 |
| superglue_multirc_FedEx-LoRA_seed44 | 20 | 0.5722 | 0.5353 | 0.5674 | 0.5716 | 0.5721 | 0.5722 |
| superglue_multirc_FedEx-LoRA_seed44 | 30 | 0.5720 | 0.5592 | 0.5719 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FedIT | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FedIT | 20 | 0.5720 | 0.5721 | 0.5721 | 0.5721 | 0.5721 | 0.5720 |
| superglue_multirc_FedIT | 30 | 0.5681 | 0.5709 | 0.5695 | 0.5687 | 0.5683 | 0.5681 |
| superglue_multirc_FedIT | 40 | 0.5720 | 0.5716 | 0.5719 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FedIT | 50 | 0.5720 | 0.5719 | 0.5721 | 0.5721 | 0.5721 | 0.5720 |
| superglue_multirc_FedIT | 60 | 0.5681 | 0.5709 | 0.5695 | 0.5687 | 0.5683 | 0.5681 |
| superglue_multirc_FedIT_seed43 | 10 | 0.5720 | 0.5586 | 0.5647 | 0.5706 | 0.5719 | 0.5720 |
| superglue_multirc_FedIT_seed43 | 20 | 0.5479 | 0.5643 | 0.5635 | 0.5591 | 0.5548 | 0.5503 |
| superglue_multirc_FedIT_seed43 | 30 | 0.5677 | 0.5654 | 0.5658 | 0.5664 | 0.5671 | 0.5676 |
| superglue_multirc_FedIT_seed43 | 40 | 0.5720 | 0.5563 | 0.5645 | 0.5706 | 0.5719 | 0.5720 |
| superglue_multirc_FedIT_seed43 | 50 | 0.5479 | 0.5635 | 0.5635 | 0.5591 | 0.5548 | 0.5503 |
| superglue_multirc_FedIT_seed43 | 60 | 0.5677 | 0.5651 | 0.5658 | 0.5664 | 0.5671 | 0.5676 |
| superglue_multirc_FedIT_seed44 | 10 | 0.5716 | 0.4855 | 0.5192 | 0.5326 | 0.5454 | 0.5614 |
| superglue_multirc_FedIT_seed44 | 20 | 0.5518 | 0.5300 | 0.5566 | 0.5569 | 0.5552 | 0.5532 |
| superglue_multirc_FedIT_seed44 | 30 | 0.5681 | 0.5512 | 0.5645 | 0.5663 | 0.5672 | 0.5678 |
| superglue_multirc_FedIT_seed44 | 40 | 0.5716 | 0.5253 | 0.5228 | 0.5327 | 0.5454 | 0.5614 |
| superglue_multirc_FedIT_seed44 | 50 | 0.5518 | 0.5439 | 0.5567 | 0.5569 | 0.5552 | 0.5532 |
| superglue_multirc_FedIT_seed44 | 60 | 0.5681 | 0.5560 | 0.5645 | 0.5663 | 0.5672 | 0.5678 |
| superglue_multirc_FlexLoRA | 10 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 | 0.5720 |
| superglue_multirc_FlexLoRA | 20 | 0.5720 | 0.5721 | 0.5721 | 0.5722 | 0.5721 | 0.5721 |
| superglue_multirc_FlexLoRA | 30 | 0.5767 | 0.5733 | 0.5750 | 0.5760 | 0.5765 | 0.5767 |
| superglue_multirc_FlexLoRA_seed43 | 10 | 0.5720 | 0.5658 | 0.5696 | 0.5722 | 0.5725 | 0.5722 |
| superglue_multirc_FlexLoRA_seed43 | 20 | 0.5761 | 0.5715 | 0.5745 | 0.5746 | 0.5748 | 0.5756 |
| superglue_multirc_FlexLoRA_seed43 | 30 | 0.5718 | 0.5714 | 0.5699 | 0.5694 | 0.5701 | 0.5712 |
| superglue_multirc_FlexLoRA_seed44 | 10 | 0.5720 | 0.4837 | 0.5155 | 0.5284 | 0.5420 | 0.5602 |
| superglue_multirc_FlexLoRA_seed44 | 20 | 0.5720 | 0.5357 | 0.5687 | 0.5718 | 0.5720 | 0.5720 |
| superglue_multirc_FlexLoRA_seed44 | 30 | 0.5720 | 0.5593 | 0.5719 | 0.5720 | 0.5720 | 0.5720 |

## Dataset: superglue_record

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| superglue_record_FFA-LoRA_seed43 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FFA-LoRA_seed43 | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FFA-LoRA_seed43 | 30 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FFA-LoRA_seed44 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FFA-LoRA_seed44 | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FL+DoRA(FlexLoRA) | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FL+DoRA(FlexLoRA) | 20 | 0.8672 | 0.8669 | 0.8675 | 0.8676 | 0.8675 | 0.8673 |
| superglue_record_FL+DoRA(FlexLoRA) | 30 | 0.8842 | 0.8756 | 0.8820 | 0.8837 | 0.8841 | 0.8842 |
| superglue_record_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FeDoRA | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FeDoRA_seed43 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FeDoRA_seed43 | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FeDoRA_seed43 | 30 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FeDoRA_seed44 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FeDoRA_seed44 | 20 | 0.8706 | 0.8670 | 0.8684 | 0.8693 | 0.8699 | 0.8704 |
| superglue_record_FeDoRA_seed44 | 30 | 0.8740 | 0.8705 | 0.8733 | 0.8739 | 0.8740 | 0.8740 |
| superglue_record_FedEx-LoRA_seed43 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedEx-LoRA_seed43 | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedEx-LoRA_seed43 | 30 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedEx-LoRA_seed44 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedEx-LoRA_seed44 | 20 | 0.8675 | 0.8666 | 0.8673 | 0.8676 | 0.8677 | 0.8676 |
| superglue_record_FedIT | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT | 30 | 0.8711 | 0.8681 | 0.8702 | 0.8708 | 0.8710 | 0.8711 |
| superglue_record_FedIT | 40 | 0.8660 | 0.8667 | 0.8661 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT | 50 | 0.8660 | 0.8663 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT | 60 | 0.8711 | 0.8682 | 0.8702 | 0.8708 | 0.8710 | 0.8711 |
| superglue_record_FedIT_seed43 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed43 | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed43 | 30 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed43 | 40 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed43 | 50 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed43 | 60 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed44 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed44 | 20 | 0.8721 | 0.8670 | 0.8686 | 0.8698 | 0.8708 | 0.8717 |
| superglue_record_FedIT_seed44 | 30 | 0.8753 | 0.8711 | 0.8744 | 0.8751 | 0.8753 | 0.8753 |
| superglue_record_FedIT_seed44 | 40 | 0.8660 | 0.8678 | 0.8662 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FedIT_seed44 | 50 | 0.8721 | 0.8676 | 0.8686 | 0.8698 | 0.8708 | 0.8717 |
| superglue_record_FedIT_seed44 | 60 | 0.8753 | 0.8713 | 0.8744 | 0.8751 | 0.8753 | 0.8753 |
| superglue_record_FlexLoRA_seed43 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FlexLoRA_seed43 | 20 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FlexLoRA_seed43 | 30 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FlexLoRA_seed44 | 10 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 | 0.8660 |
| superglue_record_FlexLoRA_seed44 | 20 | 0.8726 | 0.8677 | 0.8699 | 0.8712 | 0.8720 | 0.8724 |
| superglue_record_FlexLoRA_seed44 | 30 | 0.8779 | 0.8732 | 0.8771 | 0.8778 | 0.8779 | 0.8779 |

## Dataset: superglue_wic

| Experiment | Round | Raw Accuracy | EMA (0.1) | EMA (0.3) | EMA (0.5) | EMA (0.7) | EMA (0.9) |
|---|---|---|---|---|---|---|---|
| superglue_wic_FFA-LoRA | 10 | 0.5893 | 0.5437 | 0.5872 | 0.6010 | 0.6012 | 0.5947 |
| superglue_wic_FFA-LoRA | 20 | 0.6897 | 0.6293 | 0.6790 | 0.6850 | 0.6870 | 0.6888 |
| superglue_wic_FFA-LoRA | 30 | 0.6912 | 0.6673 | 0.6882 | 0.6895 | 0.6903 | 0.6909 |
| superglue_wic_FL+DoRA(FlexLoRA) | 10 | 0.6599 | 0.5770 | 0.6432 | 0.6601 | 0.6630 | 0.6618 |
| superglue_wic_FL+DoRA(FlexLoRA) | 20 | 0.6991 | 0.6553 | 0.6967 | 0.6987 | 0.6990 | 0.6991 |
| superglue_wic_FL+DoRA(FlexLoRA) | 30 | 0.6975 | 0.6845 | 0.6993 | 0.6985 | 0.6979 | 0.6975 |
| superglue_wic_FL+DoRA(FlexLoRA+FFALoRA) | 10 | 0.5799 | 0.5392 | 0.5805 | 0.5945 | 0.5943 | 0.5862 |
| superglue_wic_FL+DoRA(FlexLoRA+FFALoRA) | 20 | 0.6818 | 0.6216 | 0.6706 | 0.6770 | 0.6792 | 0.6810 |
| superglue_wic_FL+DoRA(FlexLoRA+FFALoRA) | 30 | 0.6834 | 0.6616 | 0.6826 | 0.6832 | 0.6833 | 0.6834 |
| superglue_wic_FeDoRA | 10 | 0.6520 | 0.5649 | 0.6274 | 0.6470 | 0.6522 | 0.6529 |
| superglue_wic_FeDoRA | 20 | 0.6959 | 0.6473 | 0.6911 | 0.6937 | 0.6944 | 0.6953 |
| superglue_wic_FeDoRA | 30 | 0.6928 | 0.6800 | 0.6959 | 0.6949 | 0.6940 | 0.6931 |
| superglue_wic_FedEx-LoRA | 10 | 0.6661 | 0.5718 | 0.6381 | 0.6603 | 0.6670 | 0.6676 |
| superglue_wic_FedEx-LoRA | 20 | 0.7006 | 0.6547 | 0.6987 | 0.7004 | 0.7001 | 0.7002 |
| superglue_wic_FedEx-LoRA | 30 | 0.6975 | 0.6818 | 0.6965 | 0.6968 | 0.6972 | 0.6975 |
| superglue_wic_FedIT | 10 | 0.6191 | 0.5533 | 0.6070 | 0.6262 | 0.6298 | 0.6246 |
| superglue_wic_FedIT | 20 | 0.6975 | 0.6423 | 0.6908 | 0.6951 | 0.6963 | 0.6972 |
| superglue_wic_FedIT | 30 | 0.6897 | 0.6736 | 0.6888 | 0.6884 | 0.6888 | 0.6894 |
| superglue_wic_FedIT | 40 | 0.6191 | 0.6139 | 0.6123 | 0.6264 | 0.6298 | 0.6246 |
| superglue_wic_FedIT | 50 | 0.6975 | 0.6634 | 0.6909 | 0.6951 | 0.6963 | 0.6972 |
| superglue_wic_FedIT | 60 | 0.6897 | 0.6810 | 0.6888 | 0.6884 | 0.6888 | 0.6894 |
| superglue_wic_FlexLoRA | 10 | 0.6599 | 0.5683 | 0.6319 | 0.6527 | 0.6596 | 0.6611 |
| superglue_wic_FlexLoRA | 20 | 0.6975 | 0.6488 | 0.6939 | 0.6973 | 0.6976 | 0.6975 |
| superglue_wic_FlexLoRA | 30 | 0.6897 | 0.6776 | 0.6918 | 0.6909 | 0.6902 | 0.6898 |

