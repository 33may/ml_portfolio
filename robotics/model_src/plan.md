### **Dataset — задачи**

* [ ] Реализовать функцию `create_sample_indices` для генерации индексов с учётом паддинга
* [ ] Реализовать функцию `sample_sequence` для выборки последовательностей с паддингом по краям
* [ ] Реализовать функцию `get_data_stats` для вычисления min/max по каждому признаку
* [ ] Реализовать функцию `normalize_data` и `unnormalize_data` для приведения данных в \[-1,1]
* [ ] Реализовать класс `PushTImageDataset`:

  * Загрузка данных из zarr
  * Нормализация `agent_pos` и `action`
  * Построение `self.indices` через `create_sample_indices`
  * Формирование одного элемента в `__getitem__` с обрезкой `obs_horizon`

---

### **Model — задачи**

* [ ] Реализовать `SinusoidalPosEmb` (позиционные эмбеддинги)
* [ ] Реализовать `Conv1dBlock`: Conv1d → GroupNorm → Mish
* [ ] Реализовать `ConditionalResidualBlock1D` с FiLM-conditioning
* [ ] Реализовать `Downsample1d` и `Upsample1d` (strided conv и transposed conv)
* [ ] Реализовать `ConditionalUnet1D`:

  * Встраивание timestep через `SinusoidalPosEmb`
  * Два residual блока в середине (`mid_modules`)
  * Стек `down_modules` c downsampling
  * Стек `up_modules` с upsampling и skip-connections
  * Финальный conv-слой (`final_conv`)

---

### **Vision encoder — задачи**

* [ ] Реализовать `get_resnet` с удалением final fc-слоя
* [ ] Реализовать `replace_bn_with_gn` через `replace_submodules` для замены BatchNorm на GroupNorm

---

###  **Scheduler — задачи**

* [ ] Интегрировать `DDPMScheduler` с cosine β-schedule
* [ ] Настроить EMA через `EMAModel`
* [ ] Настроить оптимизатор AdamW
* [ ] Настроить scheduler с warmup через `get_scheduler`
