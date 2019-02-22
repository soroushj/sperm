-- select round(100*test_acc::numeric,2) as test_acc, round(100*test_precision::numeric,2) as test_precision, round(100*test_recall::numeric,2) as test_recall, round(100*test_f05::numeric,2) as test_f05

select valid_loss, valid_acc, valid_precision, valid_f10, valid_f05, test_loss, test_acc, test_precision, test_f10, test_f05, model, flags, iter, run
from results
where label = 'a' and model = 1 and flags = 0
order by valid_loss asc, valid_acc desc, valid_precision desc
limit 1;
