# Project Direction: Therapist-in-the-Loop Musical Idea Generation

## 1. Revised problem statement

The project bottleneck is not video generation or lyric translation. During music-therapy co-composition, a therapist may temporarily run out of musical ideas while listening to a patient's story and playing live. The useful role of technology is to quickly offer several short, editable musical starting points. The therapist and patient then choose, reject, combine, and improvise on those ideas together.

The project therefore becomes:

> A heartbeat-informed, therapist-in-the-loop system that generates several short musical segment candidates for live therapist-patient co-composition.

The system is an inspiration and efficiency tool. It does not compose the final piece, interpret the patient's emotions, or replace the therapeutic relationship.

## 2. Design principles from the therapist feedback

1. **Human relationship first.** Therapeutic value comes from the patient being heard and from creating music with the therapist, not from receiving a polished machine-generated recording.
2. **Several candidates, not one answer.** The system should widen the therapist's option space and make rejection cheap.
3. **Patient material remains identifiable.** Patient-selected notes, order, and personal meaning must constrain generation rather than being replaced by an opaque model output.
4. **Therapist control remains explicit.** Key, scale, tempo, phrase length, working character, and heartbeat influence are adjustable.
5. **Outputs remain unfinished and editable.** MIDI is the main handoff format; WAV is only a quick preview.
6. **No physiological emotion diagnosis.** Heartbeat timing is embodied compositional material, not evidence of a specific emotion or clinical state.

## 3. Core users and use scenario

Primary user: a music therapist working with a patient in a co-composition session.

Typical flow:

1. The patient explains a story, image, memory, or desired meaning.
2. The patient and therapist choose a small note motif, such as scale degrees `4, 2, 7`.
3. A heartbeat recording provides a personal timing contour.
4. The therapist sets broad musical constraints.
5. The system returns four short candidates in seconds.
6. Therapist and patient audition them together.
7. They reject, combine, or improvise on selected material using piano, guitar, or a DAW.
8. The final composition remains authored through their live interaction.

Real-time generation is useful but not required for the research question. Fast turn-around during a session is the practical target.

## 4. Inputs and outputs

### Inputs

- heartbeat audio used only for BPM and IBI timing features;
- patient-selected scale degrees in a meaningful order;
- therapist controls: key, scale/mode, tempo, length, working character, and heartbeat influence;
- optional story notes shown to the therapist but not interpreted by the generator.

### Outputs

- three or four distinct musical ideas, normally 2–8 bars each;
- a quick synthesized WAV preview for auditioning;
- editable MIDI for live performance or continuation in a DAW;
- note-event CSV and a manifest explaining how each candidate was produced.

## 5. New core pipeline

```text
Heartbeat recording
  -> signal quality check
  -> BPM + beat times + IBI contour
  -> quantized personal rhythm motif --------------------+
                                                         |
Patient-selected notes -> ordered anchor motif ----------+--> constrained candidate generator
                                                         |          |
Therapist controls -> key / mode / tempo / length -------+          v
                                                       diverse short candidates
                                                         |
                                       audition -> reject / shortlist / edit
                                                         |
                                      live therapist-patient improvisation
```

The current implementation is a transparent symbolic baseline. It produces four strategies: `Pulse motif`, `Call and response`, `Expanded arc`, and `Spacious variation`. A future learned model can replace the baseline only if it preserves the same controls, traceability, latency, and editable output.

## 6. Research questions

**RQ1. Efficiency and inspiration**  
Can multiple short candidates reduce time to the first usable musical idea and help therapists recover from a creative block compared with composing from scratch?

**RQ2. Therapist agency**  
Do explicit constraints and editable multi-candidate output provide greater perceived control, trust, and usefulness than a single one-shot generated composition?

**RQ3. Patient participation**  
Does preserving patient-selected notes and their heartbeat timing motif improve perceived participation and ownership during co-composition?

The project does not claim that generated music improves anxiety, depression, or clinical outcomes. Those claims would require a separate clinical study.

## 7. Evaluation plan

Compare three conditions on the same short composition prompt:

- `Manual`: therapist starts from scratch;
- `Single`: system provides one generated idea;
- `Multi + control`: system provides several candidates with editable constraints.

Primary measures:

- time to first idea the therapist considers usable;
- candidate audition, rejection, selection, and edit counts;
- therapist-rated usefulness, controllability, diversity, and musical coherence;
- perceived interruption to conversation and live interaction;
- patient-rated participation and ownership;
- qualitative explanation of what was retained, changed, or rejected.

For an early course-project study, therapist walkthroughs and scenario-based sessions are more defensible than claims of therapeutic efficacy. Any study involving real patients requires appropriate ethics review, consent, privacy handling, and clinical supervision.

## 8. Scope boundary

### Core MVP

- heartbeat timing extraction;
- patient motif entry;
- therapist-set musical constraints;
- fast generation of several short, diverse candidates;
- side-by-side audition;
- MIDI/WAV export.

### Explicitly out of scope

- music-video generation or editing;
- lyric translation or singing-voice synthesis;
- automatic inference of emotion from heartbeat;
- autonomous full-song composition;
- automatic selection of the "best" idea on behalf of the therapist;
- selecting a production heartbeat loop;
- final recording transfer, heartbeat/song mixing, mastering, or USB delivery;
- transcription of the final live composition into printed score;
- replacing the therapist or automating the therapeutic conversation.

The last four items are real workflow needs identified by therapists, but they are separate product modules and should not dilute this project's research question.

## 9. Contribution claim

The contribution is not a new general-purpose music model. It is a focused co-creative workflow that uses personal heartbeat timing and patient-authored musical anchors to provide fast, diverse, editable inspiration while preserving therapist agency and the therapeutic relationship.
