select *
from human_scores
WHERE session_id = '3b146113-8f97-4ddc-9912-4ce86c7654cb';

WITH llm AS (
  SELECT label, SUM(strength) AS llm_total
  FROM llm_scores
  WHERE session_id = '3b146113-8f97-4ddc-9912-4ce86c7654cb'
  GROUP BY label
),
human AS (
  SELECT label, SUM(strength) AS human_total
  FROM human_scores
  WHERE session_id = '3b146113-8f97-4ddc-9912-4ce86c7654cb'
  GROUP BY label
)
SELECT
  COALESCE(llm.label, human.label) AS label,
  COALESCE(llm.llm_total, 0)      AS llm_total,
  COALESCE(human.human_total, 0)  AS human_total
FROM llm
FULL OUTER JOIN human ON llm.label = human.label
ORDER BY label;